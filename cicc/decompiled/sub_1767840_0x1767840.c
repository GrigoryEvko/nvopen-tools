// Function: sub_1767840
// Address: 0x1767840
//
__int64 __fastcall sub_1767840(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  bool v19; // al
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  bool v24; // al
  _QWORD *v25; // rsi
  bool v26; // bl
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  bool v30; // r15
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // r15
  __int64 v34; // r14
  _QWORD *v35; // rax
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 v39; // r15
  unsigned __int8 *v40; // rax
  __int64 v41; // rdi
  unsigned __int8 *v42; // rax
  __int64 v43; // rdi
  unsigned __int8 *v44; // rax
  __int64 v45; // rbx
  unsigned __int8 *v46; // rax
  bool v47; // [rsp+10h] [rbp-A8h]
  __int64 v48; // [rsp+20h] [rbp-98h] BYREF
  __int64 v49; // [rsp+28h] [rbp-90h] BYREF
  _QWORD *v50; // [rsp+30h] [rbp-88h] BYREF
  _QWORD *v51; // [rsp+38h] [rbp-80h] BYREF
  _QWORD *v52; // [rsp+40h] [rbp-78h] BYREF
  __int64 v53[2]; // [rsp+48h] [rbp-70h] BYREF
  __int16 v54; // [rsp+58h] [rbp-60h]
  __int64 v55[2]; // [rsp+68h] [rbp-50h] BYREF
  __int16 v56; // [rsp+78h] [rbp-40h]

  if ( !sub_1758E80((__int64)a1, a3, &v48, &v49, &v50, &v51, &v52) )
    return 0;
  v15 = v50;
  v16 = sub_15A37B0(*(_WORD *)(a2 + 18) & 0x7FFF, v50, a4, 0);
  v19 = sub_1596070(v16, (__int64)v15, v17, v18);
  v20 = v51;
  v47 = v19;
  v21 = sub_15A37B0(*(_WORD *)(a2 + 18) & 0x7FFF, v51, a4, 0);
  v24 = sub_1596070(v21, (__int64)v20, v22, v23);
  v25 = v52;
  v26 = v24;
  v27 = sub_15A37B0(*(_WORD *)(a2 + 18) & 0x7FFF, v52, a4, 0);
  v30 = sub_1596070(v27, (__int64)v25, v28, v29);
  v31 = sub_159C540(*(__int64 **)(a1[1] + 24));
  if ( v47 )
  {
    v43 = a1[1];
    v56 = 257;
    v54 = 257;
    v44 = sub_17203D0(v43, 40, v48, v49, v53);
    v31 = (__int64)sub_172AC10(v43, v31, (__int64)v44, v55, *(double *)a5.m128_u64, a6, a7);
    if ( !v26 )
    {
LABEL_4:
      if ( !v30 )
        goto LABEL_5;
      goto LABEL_13;
    }
  }
  else if ( !v26 )
  {
    goto LABEL_4;
  }
  v45 = a1[1];
  v56 = 257;
  v54 = 257;
  v46 = sub_17203D0(v45, 32, v48, v49, v53);
  v31 = (__int64)sub_172AC10(v45, v31, (__int64)v46, v55, *(double *)a5.m128_u64, a6, a7);
  if ( !v30 )
  {
LABEL_5:
    v32 = *(_QWORD *)(a2 + 8);
    v33 = a2;
    if ( v32 )
      goto LABEL_6;
    return 0;
  }
LABEL_13:
  v39 = a1[1];
  v54 = 257;
  v56 = 257;
  v40 = sub_17203D0(v39, 38, v48, v49, v53);
  v41 = v39;
  v33 = a2;
  v42 = sub_172AC10(v41, v31, (__int64)v40, v55, *(double *)a5.m128_u64, a6, a7);
  v32 = *(_QWORD *)(a2 + 8);
  v31 = (__int64)v42;
  if ( !v32 )
    return 0;
LABEL_6:
  v34 = *a1;
  do
  {
    v35 = sub_1648700(v32);
    sub_170B990(v34, (__int64)v35);
    v32 = *(_QWORD *)(v32 + 8);
  }
  while ( v32 );
  if ( a2 == v31 )
    v31 = sub_1599EF0(*(__int64 ***)a2);
  sub_164D160(a2, v31, a5, a6, a7, a8, v36, v37, a11, a12);
  return v33;
}
