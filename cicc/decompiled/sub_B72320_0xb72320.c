// Function: sub_B72320
// Address: 0xb72320
//
__int64 __fastcall sub_B72320(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v4; // r12
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rdi
  __int64 result; // rax
  __int64 v12; // rdi

  v2 = *(__int64 **)(a1 + 16);
  v4 = &v2[*(unsigned int *)(a1 + 24)];
  if ( v2 != v4 )
  {
    for ( i = *(_QWORD *)(a1 + 16); ; i = *(_QWORD *)(a1 + 16) )
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
  v8 = *(__int64 **)(a1 + 64);
  v9 = &v8[2 * *(unsigned int *)(a1 + 72)];
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
    v9 = *(__int64 **)(a1 + 64);
  }
  result = a1 + 80;
  if ( v9 != (__int64 *)(a1 + 80) )
    result = _libc_free(v9, a2);
  v12 = *(_QWORD *)(a1 + 16);
  if ( v12 != a1 + 32 )
    return _libc_free(v12, a2);
  return result;
}
