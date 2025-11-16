// Function: sub_1B67580
// Address: 0x1b67580
//
__int64 __fastcall sub_1B67580(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r12
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned int v22; // r12d
  _QWORD *v23; // r9
  __int64 v24; // rax
  __int64 v25; // r13
  _BOOL4 v26; // eax
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  __int64 v29; // rax
  _QWORD *v32; // [rsp+18h] [rbp-1D8h]
  _QWORD v33[4]; // [rsp+20h] [rbp-1D0h] BYREF
  _QWORD *v34; // [rsp+40h] [rbp-1B0h]
  __int64 v35; // [rsp+48h] [rbp-1A8h]
  unsigned int v36; // [rsp+50h] [rbp-1A0h]
  __int64 v37; // [rsp+58h] [rbp-198h]
  __int64 v38; // [rsp+60h] [rbp-190h]
  __int64 v39; // [rsp+68h] [rbp-188h]
  __int64 v40; // [rsp+70h] [rbp-180h]
  __int64 v41; // [rsp+78h] [rbp-178h]
  __int64 v42; // [rsp+80h] [rbp-170h]
  __int64 v43; // [rsp+88h] [rbp-168h]
  __int64 v44; // [rsp+90h] [rbp-160h]
  __int64 v45; // [rsp+98h] [rbp-158h]
  __int64 v46; // [rsp+A0h] [rbp-150h]
  __int64 v47; // [rsp+A8h] [rbp-148h]
  int v48; // [rsp+B0h] [rbp-140h]
  __int64 v49; // [rsp+B8h] [rbp-138h]
  _BYTE *v50; // [rsp+C0h] [rbp-130h]
  _BYTE *v51; // [rsp+C8h] [rbp-128h]
  __int64 v52; // [rsp+D0h] [rbp-120h]
  int v53; // [rsp+D8h] [rbp-118h]
  _BYTE v54[16]; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v55; // [rsp+F0h] [rbp-100h]
  __int64 v56; // [rsp+F8h] [rbp-F8h]
  __int64 v57; // [rsp+100h] [rbp-F0h]
  __int64 v58; // [rsp+108h] [rbp-E8h]
  __int64 v59; // [rsp+110h] [rbp-E0h]
  __int64 v60; // [rsp+118h] [rbp-D8h]
  __int16 v61; // [rsp+120h] [rbp-D0h]
  __int64 v62[5]; // [rsp+128h] [rbp-C8h] BYREF
  int v63; // [rsp+150h] [rbp-A0h]
  __int64 v64; // [rsp+158h] [rbp-98h]
  __int64 v65; // [rsp+160h] [rbp-90h]
  __int64 v66; // [rsp+168h] [rbp-88h]
  _BYTE *v67; // [rsp+170h] [rbp-80h]
  __int64 v68; // [rsp+178h] [rbp-78h]
  _BYTE v69[112]; // [rsp+180h] [rbp-70h] BYREF

  v16 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 40LL));
  v17 = *(_QWORD *)(a2 + 24);
  v33[0] = a2;
  v18 = v16;
  v33[1] = v16;
  v33[2] = "indvars";
  v33[3] = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = v54;
  v51 = v54;
  v52 = 2;
  v53 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 1;
  v19 = sub_15E0530(v17);
  v66 = v18;
  v22 = 0;
  v23 = v33;
  v62[3] = v19;
  v67 = v69;
  v68 = 0x800000000LL;
  v24 = *(_QWORD *)(a1 + 32);
  memset(v62, 0, 24);
  v62[4] = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v25 = *(_QWORD *)(*(_QWORD *)v24 + 48LL);
  while ( 1 )
  {
    if ( !v25 )
      BUG();
    if ( *(_BYTE *)(v25 - 8) != 77 )
      break;
    v32 = v23;
    v26 = sub_1B649E0((__int64 *)(v25 - 24), a2, a3, a4, a5, (__int64)v23, a6, a7, a8, a9, v20, v21, a12, a13, 0);
    v25 = *(_QWORD *)(v25 + 8);
    v23 = v32;
    v22 |= v26;
  }
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  if ( v62[0] )
    sub_161E7C0((__int64)v62, v62[0]);
  j___libc_free_0(v58);
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  j___libc_free_0(v46);
  j___libc_free_0(v42);
  j___libc_free_0(v38);
  if ( v36 )
  {
    v27 = v34;
    v28 = &v34[5 * v36];
    do
    {
      while ( *v27 == -8 )
      {
        if ( v27[1] != -8 )
          goto LABEL_14;
        v27 += 5;
        if ( v28 == v27 )
          goto LABEL_21;
      }
      if ( *v27 != -16 || v27[1] != -16 )
      {
LABEL_14:
        v29 = v27[4];
        if ( v29 != 0 && v29 != -8 && v29 != -16 )
          sub_1649B30(v27 + 2);
      }
      v27 += 5;
    }
    while ( v28 != v27 );
  }
LABEL_21:
  j___libc_free_0(v34);
  return v22;
}
