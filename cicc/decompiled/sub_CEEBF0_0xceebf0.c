// Function: sub_CEEBF0
// Address: 0xceebf0
//
void __fastcall sub_CEEBF0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v4; // r12
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rdi
  _QWORD *v11; // rdi

  v2 = (__int64 *)a1[2];
  v4 = &v2[*((unsigned int *)a1 + 6)];
  if ( v2 != v4 )
  {
    for ( i = a1[2]; ; i = a1[2] )
    {
      v6 = *v2;
      v7 = (unsigned int)(((__int64)v2 - i) >> 3) >> 7;
      a2 = 4096LL << v7;
      if ( v7 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v2;
      sub_C7D6A0(v6, a2, 16);
      if ( v4 == v2 )
        break;
    }
  }
  v8 = (__int64 *)a1[8];
  v9 = &v8[2 * *((unsigned int *)a1 + 18)];
  if ( v8 != v9 )
  {
    do
    {
      a2 = v8[1];
      v10 = *v8;
      v8 += 2;
      sub_C7D6A0(v10, a2, 16);
    }
    while ( v9 != v8 );
    v9 = (__int64 *)a1[8];
  }
  if ( v9 != a1 + 10 )
    _libc_free(v9, a2);
  v11 = (_QWORD *)a1[2];
  if ( v11 != a1 + 4 )
    _libc_free(v11, a2);
}
