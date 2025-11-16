// Function: sub_2CE96D0
// Address: 0x2ce96d0
//
__int64 __fastcall sub_2CE96D0(__int64 a1, __int64 a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v13; // [rsp+0h] [rbp-2A0h] BYREF
  int v14; // [rsp+8h] [rbp-298h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-290h]
  int *v16; // [rsp+18h] [rbp-288h]
  int *v17; // [rsp+20h] [rbp-280h]
  __int64 v18; // [rsp+28h] [rbp-278h]
  __int64 v19; // [rsp+30h] [rbp-270h] BYREF
  int v20; // [rsp+38h] [rbp-268h] BYREF
  unsigned __int64 v21; // [rsp+40h] [rbp-260h]
  int *v22; // [rsp+48h] [rbp-258h]
  int *v23; // [rsp+50h] [rbp-250h]
  __int64 v24; // [rsp+58h] [rbp-248h]
  _QWORD v25[19]; // [rsp+60h] [rbp-240h] BYREF
  int v26; // [rsp+F8h] [rbp-1A8h]
  __int64 v27; // [rsp+100h] [rbp-1A0h]
  __int64 v28; // [rsp+108h] [rbp-198h] BYREF
  int v29; // [rsp+118h] [rbp-188h] BYREF
  __int64 v30; // [rsp+120h] [rbp-180h]
  int *v31; // [rsp+128h] [rbp-178h]
  int *v32; // [rsp+130h] [rbp-170h]
  __int64 v33; // [rsp+138h] [rbp-168h]
  __int64 v34; // [rsp+140h] [rbp-160h]
  __int64 v35; // [rsp+148h] [rbp-158h]
  __int64 v36; // [rsp+150h] [rbp-150h]
  int v37; // [rsp+158h] [rbp-148h]
  int v38; // [rsp+168h] [rbp-138h] BYREF
  __int64 v39; // [rsp+170h] [rbp-130h]
  int *v40; // [rsp+178h] [rbp-128h]
  int *v41; // [rsp+180h] [rbp-120h]
  __int64 v42; // [rsp+188h] [rbp-118h]
  int v43; // [rsp+198h] [rbp-108h] BYREF
  __int64 v44; // [rsp+1A0h] [rbp-100h]
  int *v45; // [rsp+1A8h] [rbp-F8h]
  int *v46; // [rsp+1B0h] [rbp-F0h]
  __int64 v47; // [rsp+1B8h] [rbp-E8h]
  __int64 v48; // [rsp+1C0h] [rbp-E0h]
  __int64 v49; // [rsp+1C8h] [rbp-D8h]
  __int64 v50; // [rsp+1D0h] [rbp-D0h]
  int v51; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v52; // [rsp+1E8h] [rbp-B8h]
  int *v53; // [rsp+1F0h] [rbp-B0h]
  int *v54; // [rsp+1F8h] [rbp-A8h]
  __int64 v55; // [rsp+200h] [rbp-A0h]
  __int64 v56; // [rsp+208h] [rbp-98h]
  __int64 v57; // [rsp+210h] [rbp-90h]
  __int64 v58; // [rsp+218h] [rbp-88h]
  int v59; // [rsp+228h] [rbp-78h] BYREF
  __int64 v60; // [rsp+230h] [rbp-70h]
  int *v61; // [rsp+238h] [rbp-68h]
  int *v62; // [rsp+240h] [rbp-60h]
  __int64 v63; // [rsp+248h] [rbp-58h]
  int v64; // [rsp+258h] [rbp-48h] BYREF
  __int64 v65; // [rsp+260h] [rbp-40h]
  int *v66; // [rsp+268h] [rbp-38h]
  int *v67; // [rsp+270h] [rbp-30h]
  __int64 v68; // [rsp+278h] [rbp-28h]

  LODWORD(v6) = 1;
  v16 = &v14;
  v17 = &v14;
  v22 = &v20;
  v23 = &v20;
  v7 = *(_QWORD *)(a2 + 8);
  v14 = 0;
  v15 = 0;
  v18 = 0;
  v20 = 0;
  v21 = 0;
  v24 = 0;
  LODWORD(v7) = *(_DWORD *)(v7 + 8) >> 8;
  *a3 = v7;
  if ( !(_DWORD)v7 )
  {
    v25[1] = a6;
    v6 = &v13;
    v25[15] = &v28;
    v31 = &v29;
    v32 = &v29;
    v25[2] = a5;
    v25[0] = 256;
    memset(&v25[3], 0, 96);
    v25[16] = 1;
    v25[17] = 0;
    v25[18] = 0;
    v26 = 1065353216;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = &v38;
    v41 = &v38;
    v45 = &v43;
    v46 = &v43;
    v53 = &v51;
    v54 = &v51;
    v48 = a4;
    v61 = &v59;
    v62 = &v59;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v47 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = &v64;
    v67 = &v64;
    v68 = 0;
    LOBYTE(v6) = (unsigned int)sub_2CE8530((__int64)v25, a1, (unsigned __int8 *)a2, &v13, &v19, a3) == 1;
    sub_2CE0050((__int64)v25);
    v9 = v21;
    while ( v9 )
    {
      sub_2CDF380(*(_QWORD *)(v9 + 24));
      v10 = v9;
      v9 = *(_QWORD *)(v9 + 16);
      j_j___libc_free_0(v10);
    }
    v11 = v15;
    while ( v11 )
    {
      sub_2CDE640(*(_QWORD *)(v11 + 24));
      v12 = v11;
      v11 = *(_QWORD *)(v11 + 16);
      j_j___libc_free_0(v12);
    }
  }
  return (unsigned int)v6;
}
