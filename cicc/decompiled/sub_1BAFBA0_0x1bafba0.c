// Function: sub_1BAFBA0
// Address: 0x1bafba0
//
__int64 __fastcall sub_1BAFBA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  double v22; // xmm4_8
  double v23; // xmm5_8
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  unsigned __int64 v26; // rdi
  void *v28; // [rsp+0h] [rbp-110h] BYREF
  __int64 v29; // [rsp+8h] [rbp-108h]
  __int64 v30; // [rsp+10h] [rbp-100h] BYREF
  char v31; // [rsp+20h] [rbp-F0h]
  __int64 v32; // [rsp+28h] [rbp-E8h]
  _QWORD *v33; // [rsp+30h] [rbp-E0h]
  __int64 v34; // [rsp+38h] [rbp-D8h]
  unsigned int v35; // [rsp+40h] [rbp-D0h]
  __int64 v36; // [rsp+48h] [rbp-C8h]
  __int64 v37; // [rsp+50h] [rbp-C0h]
  __int64 v38; // [rsp+58h] [rbp-B8h]
  __int64 v39; // [rsp+60h] [rbp-B0h]
  __int64 v40; // [rsp+68h] [rbp-A8h]
  __int64 v41; // [rsp+70h] [rbp-A0h] BYREF
  _QWORD v42[5]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+D8h] [rbp-38h]
  __int64 v44; // [rsp+E0h] [rbp-30h]
  int v45; // [rsp+E8h] [rbp-28h]
  __int64 v46; // [rsp+F0h] [rbp-20h]
  void **v47; // [rsp+F8h] [rbp-18h]

  v13 = *(_QWORD *)(a1 + 8);
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v28 = &unk_49F6E90;
  v14 = *(_QWORD *)(a1 + 112);
  v15 = a2 + 96;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 1;
  v29 = a2;
  v16 = a2 + 280;
  v30 = v14;
  v17 = &v41;
  do
  {
    *v17 = -8;
    v17 += 2;
  }
  while ( v17 != v42 );
  v42[2] = v15;
  v42[0] = v13;
  v42[1] = a3;
  v42[3] = v16;
  v42[4] = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = a2;
  v47 = &v28;
  v37 = sub_1BADF80(a2, a4, a5, *(double *)a6.m128i_i64, a7, a8, a9, a10, a11);
  *(double *)a4.m128i_i64 = ((double (__fastcall *)(_QWORD, __int64 *, __int64, __int64, __int64, __int64, void *, __int64))sub_1BE6920)(
                              **(_QWORD **)(a1 + 48),
                              &v30,
                              v18,
                              v19,
                              v20,
                              v21,
                              v28,
                              v29);
  sub_1BAF4D0(a2, (__m128)a4, a5, a6, a7, v22, v23, a10, a11);
  j___libc_free_0(v43);
  if ( (v40 & 1) == 0 )
    j___libc_free_0(v41);
  if ( v35 )
  {
    v24 = v33;
    v25 = &v33[5 * v35];
    do
    {
      if ( *v24 != -8 && *v24 != -16 )
      {
        v26 = v24[1];
        if ( (_QWORD *)v26 != v24 + 3 )
          _libc_free(v26);
      }
      v24 += 5;
    }
    while ( v25 != v24 );
  }
  return j___libc_free_0(v33);
}
