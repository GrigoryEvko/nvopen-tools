// Function: sub_1CA0940
// Address: 0x1ca0940
//
__int64 __fastcall sub_1CA0940(
        __int64 a1,
        __int64 *a2,
        int *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-2A0h] BYREF
  int v22; // [rsp+8h] [rbp-298h] BYREF
  __int64 v23; // [rsp+10h] [rbp-290h]
  int *v24; // [rsp+18h] [rbp-288h]
  int *v25; // [rsp+20h] [rbp-280h]
  __int64 v26; // [rsp+28h] [rbp-278h]
  __int64 v27; // [rsp+30h] [rbp-270h] BYREF
  int v28; // [rsp+38h] [rbp-268h] BYREF
  __int64 v29; // [rsp+40h] [rbp-260h]
  int *v30; // [rsp+48h] [rbp-258h]
  int *v31; // [rsp+50h] [rbp-250h]
  __int64 v32; // [rsp+58h] [rbp-248h]
  _QWORD v33[19]; // [rsp+60h] [rbp-240h] BYREF
  int v34; // [rsp+F8h] [rbp-1A8h]
  __int64 v35; // [rsp+100h] [rbp-1A0h]
  __int64 v36; // [rsp+108h] [rbp-198h] BYREF
  int v37; // [rsp+118h] [rbp-188h] BYREF
  __int64 v38; // [rsp+120h] [rbp-180h]
  int *v39; // [rsp+128h] [rbp-178h]
  int *v40; // [rsp+130h] [rbp-170h]
  __int64 v41; // [rsp+138h] [rbp-168h]
  __int64 v42; // [rsp+140h] [rbp-160h]
  __int64 v43; // [rsp+148h] [rbp-158h]
  __int64 v44; // [rsp+150h] [rbp-150h]
  int v45; // [rsp+158h] [rbp-148h]
  int v46; // [rsp+168h] [rbp-138h] BYREF
  __int64 v47; // [rsp+170h] [rbp-130h]
  int *v48; // [rsp+178h] [rbp-128h]
  int *v49; // [rsp+180h] [rbp-120h]
  __int64 v50; // [rsp+188h] [rbp-118h]
  int v51; // [rsp+198h] [rbp-108h] BYREF
  __int64 v52; // [rsp+1A0h] [rbp-100h]
  int *v53; // [rsp+1A8h] [rbp-F8h]
  int *v54; // [rsp+1B0h] [rbp-F0h]
  __int64 v55; // [rsp+1B8h] [rbp-E8h]
  __int64 v56; // [rsp+1C0h] [rbp-E0h]
  __int64 v57; // [rsp+1C8h] [rbp-D8h]
  __int64 v58; // [rsp+1D0h] [rbp-D0h]
  int v59; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+1E8h] [rbp-B8h]
  int *v61; // [rsp+1F0h] [rbp-B0h]
  int *v62; // [rsp+1F8h] [rbp-A8h]
  __int64 v63; // [rsp+200h] [rbp-A0h]
  __int64 v64; // [rsp+208h] [rbp-98h]
  __int64 v65; // [rsp+210h] [rbp-90h]
  __int64 v66; // [rsp+218h] [rbp-88h]
  int v67; // [rsp+228h] [rbp-78h] BYREF
  __int64 v68; // [rsp+230h] [rbp-70h]
  int *v69; // [rsp+238h] [rbp-68h]
  int *v70; // [rsp+240h] [rbp-60h]
  __int64 v71; // [rsp+248h] [rbp-58h]
  int v72; // [rsp+258h] [rbp-48h] BYREF
  __int64 v73; // [rsp+260h] [rbp-40h]
  int *v74; // [rsp+268h] [rbp-38h]
  int *v75; // [rsp+270h] [rbp-30h]
  __int64 v76; // [rsp+278h] [rbp-28h]

  LODWORD(v14) = 1;
  v24 = &v22;
  v25 = &v22;
  v30 = &v28;
  v31 = &v28;
  v15 = *a2;
  v22 = 0;
  v23 = 0;
  v26 = 0;
  v28 = 0;
  v29 = 0;
  v32 = 0;
  LODWORD(v15) = *(_DWORD *)(v15 + 8) >> 8;
  *a3 = v15;
  if ( !(_DWORD)v15 )
  {
    v33[1] = a6;
    v14 = &v21;
    v33[15] = &v36;
    v39 = &v37;
    v40 = &v37;
    v33[2] = a5;
    v33[0] = 256;
    memset(&v33[3], 0, 96);
    v33[16] = 1;
    v33[17] = 0;
    v33[18] = 0;
    v34 = 1065353216;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = &v46;
    v49 = &v46;
    v53 = &v51;
    v54 = &v51;
    v61 = &v59;
    v62 = &v59;
    v56 = a4;
    v69 = &v67;
    v70 = &v67;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    v55 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = &v72;
    v75 = &v72;
    v76 = 0;
    LOBYTE(v14) = (unsigned int)sub_1C9F820(
                                  (__int64)v33,
                                  a1,
                                  (unsigned __int64)a2,
                                  &v21,
                                  &v27,
                                  a3,
                                  a7,
                                  a8,
                                  a9,
                                  a10,
                                  a11,
                                  a12,
                                  a13,
                                  a14) == 1;
    sub_1C985C0((__int64)v33);
    v17 = v29;
    while ( v17 )
    {
      sub_1C97470(*(_QWORD *)(v17 + 24));
      v18 = v17;
      v17 = *(_QWORD *)(v17 + 16);
      j_j___libc_free_0(v18, 40);
    }
    v19 = v23;
    while ( v19 )
    {
      sub_1C96570(*(_QWORD *)(v19 + 24));
      v20 = v19;
      v19 = *(_QWORD *)(v19 + 16);
      j_j___libc_free_0(v20, 48);
    }
  }
  return (unsigned int)v14;
}
