// Function: sub_388D240
// Address: 0x388d240
//
void __fastcall sub_388D240(
        _QWORD *a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  __int64 v11; // r13
  __int64 v12; // rsi
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 v15; // rdx
  __int64 v16; // rcx
  double v17; // xmm4_8
  double v18; // xmm5_8
  __int64 i; // r12
  __int64 v20; // r13
  __int64 v21; // rsi
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // rdx
  __int64 v25; // rcx
  double v26; // xmm4_8
  double v27; // xmm5_8
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi

  v9 = a1[5];
  if ( (_QWORD *)v9 != a1 + 3 )
  {
    do
    {
      v11 = *(_QWORD *)(v9 + 64);
      if ( *(_BYTE *)(v11 + 16) != 18 )
      {
        v12 = sub_1599EF0(*(__int64 ***)v11);
        sub_164D160(v11, v12, a2, a3, a4, a5, v13, v14, a8, a9);
        sub_164BEC0(*(_QWORD *)(v9 + 64), v12, v15, v16, a2, a3, a4, a5, v17, v18, a8, a9);
      }
      v9 = sub_220EEE0(v9);
    }
    while ( a1 + 3 != (_QWORD *)v9 );
  }
  for ( i = a1[11]; a1 + 9 != (_QWORD *)i; i = sub_220EEE0(i) )
  {
    v20 = *(_QWORD *)(i + 40);
    if ( *(_BYTE *)(v20 + 16) != 18 )
    {
      v21 = sub_1599EF0(*(__int64 ***)v20);
      sub_164D160(v20, v21, a2, a3, a4, a5, v22, v23, a8, a9);
      sub_164BEC0(*(_QWORD *)(i + 40), v21, v24, v25, a2, a3, a4, a5, v26, v27, a8, a9);
    }
  }
  v28 = a1[14];
  if ( v28 )
    j_j___libc_free_0(v28);
  v29 = a1[10];
  while ( v29 )
  {
    sub_3887900(*(_QWORD *)(v29 + 24));
    v30 = v29;
    v29 = *(_QWORD *)(v29 + 16);
    j_j___libc_free_0(v30);
  }
  v31 = a1[4];
  while ( v31 )
  {
    v32 = v31;
    sub_38882E0(*(_QWORD **)(v31 + 24));
    v33 = *(_QWORD *)(v31 + 32);
    v31 = *(_QWORD *)(v31 + 16);
    if ( v33 != v32 + 48 )
      j_j___libc_free_0(v33);
    j_j___libc_free_0(v32);
  }
}
