// Function: sub_2EA58D0
// Address: 0x2ea58d0
//
__int64 __fastcall sub_2EA58D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 *v10; // rbx
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  v3 = a1 + 200;
  *(_QWORD *)(v3 - 200) = &unk_4A29558;
  sub_2EA54C0(v3, a2);
  v4 = *(__int64 **)(a1 + 272);
  v5 = &v4[*(unsigned int *)(a1 + 280)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 272); ; i = *(_QWORD *)(a1 + 272) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      v9 = 4096LL << v8;
      if ( v8 >= 0x1E )
        v9 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, v9, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v10 = *(__int64 **)(a1 + 320);
  v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(a1 + 328)];
  if ( v10 != (__int64 *)v11 )
  {
    do
    {
      v12 = v10[1];
      v13 = *v10;
      v10 += 2;
      sub_C7D6A0(v13, v12, 16);
    }
    while ( (__int64 *)v11 != v10 );
    v11 = *(_QWORD *)(a1 + 320);
  }
  if ( v11 != a1 + 336 )
    _libc_free(v11);
  v14 = *(_QWORD *)(a1 + 272);
  if ( v14 != a1 + 288 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 232);
  if ( v15 )
    j_j___libc_free_0(v15);
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 16LL * *(unsigned int *)(a1 + 224), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
