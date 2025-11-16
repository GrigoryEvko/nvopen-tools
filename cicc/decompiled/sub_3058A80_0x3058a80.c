// Function: sub_3058A80
// Address: 0x3058a80
//
__int64 __fastcall sub_3058A80(__int64 a1)
{
  __int64 v2; // rsi
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

  v2 = *(unsigned int *)(a1 + 496);
  *(_QWORD *)a1 = &unk_4A2F570;
  sub_C7D6A0(*(_QWORD *)(a1 + 480), 16 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 448), 16LL * *(unsigned int *)(a1 + 464), 8);
  v3 = *(__int64 **)(a1 + 352);
  v4 = &v3[*(unsigned int *)(a1 + 360)];
  if ( v3 != v4 )
  {
    for ( i = *(_QWORD *)(a1 + 352); ; i = *(_QWORD *)(a1 + 352) )
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
  v9 = *(__int64 **)(a1 + 400);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 408)];
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
    v10 = *(_QWORD *)(a1 + 400);
  }
  if ( v10 != a1 + 416 )
    _libc_free(v10);
  v13 = *(_QWORD *)(a1 + 352);
  if ( v13 != a1 + 368 )
    _libc_free(v13);
  *(_QWORD *)a1 = &unk_4A2F290;
  return sub_2FF61F0(a1);
}
