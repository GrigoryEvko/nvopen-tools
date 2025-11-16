// Function: sub_1CACAA0
// Address: 0x1cacaa0
//
__int64 __fastcall sub_1CACAA0(
        __int16 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned int v13; // r12d
  __int64 v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int16 v18; // dx
  int v19; // r12d
  double v20; // xmm4_8
  double v21; // xmm5_8
  _QWORD v23[19]; // [rsp+10h] [rbp-250h] BYREF
  int v24; // [rsp+A8h] [rbp-1B8h]
  __int64 v25; // [rsp+B0h] [rbp-1B0h]
  __int64 v26; // [rsp+B8h] [rbp-1A8h] BYREF
  int v27; // [rsp+C8h] [rbp-198h] BYREF
  __int64 v28; // [rsp+D0h] [rbp-190h]
  int *v29; // [rsp+D8h] [rbp-188h]
  int *v30; // [rsp+E0h] [rbp-180h]
  __int64 v31; // [rsp+E8h] [rbp-178h]
  __int64 v32; // [rsp+F0h] [rbp-170h]
  __int64 v33; // [rsp+F8h] [rbp-168h]
  __int64 v34; // [rsp+100h] [rbp-160h]
  int v35; // [rsp+108h] [rbp-158h]
  int v36; // [rsp+118h] [rbp-148h] BYREF
  __int64 v37; // [rsp+120h] [rbp-140h]
  int *v38; // [rsp+128h] [rbp-138h]
  int *v39; // [rsp+130h] [rbp-130h]
  __int64 v40; // [rsp+138h] [rbp-128h]
  int v41; // [rsp+148h] [rbp-118h] BYREF
  __int64 v42; // [rsp+150h] [rbp-110h]
  int *v43; // [rsp+158h] [rbp-108h]
  int *v44; // [rsp+160h] [rbp-100h]
  __int64 v45; // [rsp+168h] [rbp-F8h]
  __int64 v46; // [rsp+170h] [rbp-F0h]
  __int64 v47; // [rsp+178h] [rbp-E8h]
  __int64 v48; // [rsp+180h] [rbp-E0h]
  int v49; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v50; // [rsp+198h] [rbp-C8h]
  int *v51; // [rsp+1A0h] [rbp-C0h]
  int *v52; // [rsp+1A8h] [rbp-B8h]
  __int64 v53; // [rsp+1B0h] [rbp-B0h]
  __int64 v54; // [rsp+1B8h] [rbp-A8h]
  __int64 v55; // [rsp+1C0h] [rbp-A0h]
  __int64 v56; // [rsp+1C8h] [rbp-98h]
  int v57; // [rsp+1D8h] [rbp-88h] BYREF
  __int64 v58; // [rsp+1E0h] [rbp-80h]
  int *v59; // [rsp+1E8h] [rbp-78h]
  int *v60; // [rsp+1F0h] [rbp-70h]
  __int64 v61; // [rsp+1F8h] [rbp-68h]
  int v62; // [rsp+208h] [rbp-58h] BYREF
  __int64 v63; // [rsp+210h] [rbp-50h]
  int *v64; // [rsp+218h] [rbp-48h]
  int *v65; // [rsp+220h] [rbp-40h]
  __int64 v66; // [rsp+228h] [rbp-38h]

  v13 = 0;
  v15 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  if ( v15 )
  {
    v18 = *a1;
    memset(&v23[1], 0, 112);
    LOWORD(v23[0]) = v18;
    v23[15] = &v26;
    v29 = &v27;
    v30 = &v27;
    v23[16] = 1;
    v23[17] = 0;
    v23[18] = 0;
    v24 = 1065353216;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v46 = v15;
    v51 = &v49;
    v52 = &v49;
    v38 = &v36;
    v39 = &v36;
    v59 = &v57;
    v60 = &v57;
    v43 = &v41;
    v44 = &v41;
    v37 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v45 = 0;
    v47 = a3;
    v48 = a4;
    v49 = 0;
    v50 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = &v62;
    v65 = &v62;
    v66 = 0;
    v19 = sub_1CAC930(v23, a2, a5, a6, a7, a8, v16, v17, a11, a12);
    v13 = sub_1CA75A0((__int64)a1, a2, a5, a6, a7, a8, v20, v21, a11, a12) | v19;
    sub_1C985C0((__int64)v23);
  }
  return v13;
}
