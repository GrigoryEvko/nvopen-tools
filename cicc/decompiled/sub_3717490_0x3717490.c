// Function: sub_3717490
// Address: 0x3717490
//
void __fastcall sub_3717490(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  *(_QWORD *)a1 = &unk_4A3CB08;
  v2 = *(_QWORD *)(a1 + 112);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(__int64 **)(a1 + 24);
  v4 = &v3[*(unsigned int *)(a1 + 32)];
  if ( v3 != v4 )
  {
    for ( i = *(_QWORD *)(a1 + 24); ; i = *(_QWORD *)(a1 + 24) )
    {
      v6 = *v3;
      v7 = (unsigned int)(((__int64)v3 - i) >> 3) >> 7;
      v8 = 4096LL << v7;
      if ( v7 >= 0x1E )
        v8 = 0x40000000000LL;
      ++v3;
      sub_C7D6A0(v6, v8, 16);
      if ( v4 == v3 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 72);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 80)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      v11 = v9[1];
      v12 = *v9;
      v9 += 2;
      sub_C7D6A0(v12, v11, 16);
    }
    while ( (__int64 *)v10 != v9 );
    v10 = *(_QWORD *)(a1 + 72);
  }
  if ( v10 != a1 + 88 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 != a1 + 40 )
    _libc_free(v13);
  j_j___libc_free_0(a1);
}
