// Function: sub_36D55B0
// Address: 0x36d55b0
//
void __fastcall sub_36D55B0(unsigned __int64 a1)
{
  __int64 v1; // r14
  __int64 v3; // rsi
  __int64 *v4; // r15
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

  v1 = a1 + 80;
  v3 = 16LL * *(unsigned int *)(a1 + 576);
  *(_QWORD *)a1 = &unk_4A3B778;
  *(_QWORD *)(a1 + 80) = &unk_4A2F570;
  sub_C7D6A0(*(_QWORD *)(a1 + 560), v3, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 528), 16LL * *(unsigned int *)(a1 + 544), 8);
  v4 = *(__int64 **)(a1 + 432);
  v5 = &v4[*(unsigned int *)(a1 + 440)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 432); ; i = *(_QWORD *)(a1 + 432) )
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
  v10 = *(__int64 **)(a1 + 480);
  v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(a1 + 488)];
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
    v11 = *(_QWORD *)(a1 + 480);
  }
  if ( v11 != a1 + 496 )
    _libc_free(v11);
  v14 = *(_QWORD *)(a1 + 432);
  if ( v14 != a1 + 448 )
    _libc_free(v14);
  *(_QWORD *)(a1 + 80) = &unk_4A2F290;
  sub_2FF61F0(v1);
  *(_QWORD *)a1 = &unk_4A2FD98;
  sub_2FDD4A0((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
