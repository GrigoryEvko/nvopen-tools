// Function: sub_1CACE90
// Address: 0x1cace90
//
__int64 __fastcall sub_1CACE90(
        __int64 a1,
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
  unsigned int v12; // r12d
  _QWORD v14[19]; // [rsp+0h] [rbp-230h] BYREF
  int v15; // [rsp+98h] [rbp-198h]
  __int64 v16; // [rsp+A0h] [rbp-190h]
  __int64 v17; // [rsp+A8h] [rbp-188h] BYREF
  int v18; // [rsp+B8h] [rbp-178h] BYREF
  __int64 v19; // [rsp+C0h] [rbp-170h]
  int *v20; // [rsp+C8h] [rbp-168h]
  int *v21; // [rsp+D0h] [rbp-160h]
  __int64 v22; // [rsp+D8h] [rbp-158h]
  __int64 v23; // [rsp+E0h] [rbp-150h]
  __int64 v24; // [rsp+E8h] [rbp-148h]
  __int64 v25; // [rsp+F0h] [rbp-140h]
  int v26; // [rsp+F8h] [rbp-138h]
  int v27; // [rsp+108h] [rbp-128h] BYREF
  __int64 v28; // [rsp+110h] [rbp-120h]
  int *v29; // [rsp+118h] [rbp-118h]
  int *v30; // [rsp+120h] [rbp-110h]
  __int64 v31; // [rsp+128h] [rbp-108h]
  int v32; // [rsp+138h] [rbp-F8h] BYREF
  __int64 v33; // [rsp+140h] [rbp-F0h]
  int *v34; // [rsp+148h] [rbp-E8h]
  int *v35; // [rsp+150h] [rbp-E0h]
  __int64 v36; // [rsp+158h] [rbp-D8h]
  __int64 v37; // [rsp+160h] [rbp-D0h]
  __int64 v38; // [rsp+168h] [rbp-C8h]
  __int64 v39; // [rsp+170h] [rbp-C0h]
  int v40; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+188h] [rbp-A8h]
  int *v42; // [rsp+190h] [rbp-A0h]
  int *v43; // [rsp+198h] [rbp-98h]
  __int64 v44; // [rsp+1A0h] [rbp-90h]
  __int64 v45; // [rsp+1A8h] [rbp-88h]
  __int64 v46; // [rsp+1B0h] [rbp-80h]
  __int64 v47; // [rsp+1B8h] [rbp-78h]
  int v48; // [rsp+1C8h] [rbp-68h] BYREF
  __int64 v49; // [rsp+1D0h] [rbp-60h]
  int *v50; // [rsp+1D8h] [rbp-58h]
  int *v51; // [rsp+1E0h] [rbp-50h]
  __int64 v52; // [rsp+1E8h] [rbp-48h]
  int v53; // [rsp+1F8h] [rbp-38h] BYREF
  __int64 v54; // [rsp+200h] [rbp-30h]
  int *v55; // [rsp+208h] [rbp-28h]
  int *v56; // [rsp+210h] [rbp-20h]
  __int64 v57; // [rsp+218h] [rbp-18h]

  v14[15] = &v17;
  v20 = &v18;
  v21 = &v18;
  v14[1] = a4;
  v14[2] = a3;
  v14[0] = 256;
  memset(&v14[3], 0, 96);
  v14[16] = 1;
  v14[17] = 0;
  v14[18] = 0;
  v15 = 1065353216;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v29 = &v27;
  v30 = &v27;
  v34 = &v32;
  v35 = &v32;
  v42 = &v40;
  v43 = &v40;
  v37 = a2;
  v50 = &v48;
  v51 = &v48;
  v55 = &v53;
  v56 = &v53;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v36 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v57 = 0;
  v12 = sub_1CAC930(v14, a1, a5, a6, a7, a8, a9, a10, a11, a12);
  sub_1C985C0((__int64)v14);
  return v12;
}
