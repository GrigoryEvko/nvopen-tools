// Function: sub_2CF5350
// Address: 0x2cf5350
//
__int64 __fastcall sub_2CF5350(__int16 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v5; // dx
  __int64 v6; // rax
  int v7; // ebx
  unsigned int v8; // ebx
  _QWORD v10[19]; // [rsp+0h] [rbp-240h] BYREF
  int v11; // [rsp+98h] [rbp-1A8h]
  __int64 v12; // [rsp+A0h] [rbp-1A0h]
  __int64 v13; // [rsp+A8h] [rbp-198h] BYREF
  int v14; // [rsp+B8h] [rbp-188h] BYREF
  __int64 v15; // [rsp+C0h] [rbp-180h]
  int *v16; // [rsp+C8h] [rbp-178h]
  int *v17; // [rsp+D0h] [rbp-170h]
  __int64 v18; // [rsp+D8h] [rbp-168h]
  __int64 v19; // [rsp+E0h] [rbp-160h]
  __int64 v20; // [rsp+E8h] [rbp-158h]
  __int64 v21; // [rsp+F0h] [rbp-150h]
  int v22; // [rsp+F8h] [rbp-148h]
  int v23; // [rsp+108h] [rbp-138h] BYREF
  __int64 v24; // [rsp+110h] [rbp-130h]
  int *v25; // [rsp+118h] [rbp-128h]
  int *v26; // [rsp+120h] [rbp-120h]
  __int64 v27; // [rsp+128h] [rbp-118h]
  int v28; // [rsp+138h] [rbp-108h] BYREF
  __int64 v29; // [rsp+140h] [rbp-100h]
  int *v30; // [rsp+148h] [rbp-F8h]
  int *v31; // [rsp+150h] [rbp-F0h]
  __int64 v32; // [rsp+158h] [rbp-E8h]
  __int64 v33; // [rsp+160h] [rbp-E0h]
  __int64 v34; // [rsp+168h] [rbp-D8h]
  __int64 v35; // [rsp+170h] [rbp-D0h]
  int v36; // [rsp+180h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+188h] [rbp-B8h]
  int *v38; // [rsp+190h] [rbp-B0h]
  int *v39; // [rsp+198h] [rbp-A8h]
  __int64 v40; // [rsp+1A0h] [rbp-A0h]
  __int64 v41; // [rsp+1A8h] [rbp-98h]
  __int64 v42; // [rsp+1B0h] [rbp-90h]
  __int64 v43; // [rsp+1B8h] [rbp-88h]
  int v44; // [rsp+1C8h] [rbp-78h] BYREF
  __int64 v45; // [rsp+1D0h] [rbp-70h]
  int *v46; // [rsp+1D8h] [rbp-68h]
  int *v47; // [rsp+1E0h] [rbp-60h]
  __int64 v48; // [rsp+1E8h] [rbp-58h]
  int v49; // [rsp+1F8h] [rbp-48h] BYREF
  __int64 v50; // [rsp+200h] [rbp-40h]
  int *v51; // [rsp+208h] [rbp-38h]
  int *v52; // [rsp+210h] [rbp-30h]
  __int64 v53; // [rsp+218h] [rbp-28h]

  v5 = *a1;
  v6 = *(_QWORD *)(a2 + 40);
  memset(&v10[1], 0, 112);
  LOWORD(v10[0]) = v5;
  v10[15] = &v13;
  v10[16] = 1;
  v10[17] = 0;
  v10[18] = 0;
  v11 = 1065353216;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = &v14;
  v33 = v6 + 312;
  v17 = &v14;
  v38 = &v36;
  v39 = &v36;
  v25 = &v23;
  v26 = &v23;
  v46 = &v44;
  v47 = &v44;
  v30 = &v28;
  v31 = &v28;
  v34 = a3;
  v35 = a4;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v32 = 0;
  v36 = 0;
  v37 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = &v49;
  v52 = &v49;
  v53 = 0;
  v7 = sub_2CF51E0(v10, a2);
  v8 = sub_2CF2660((__int64)a1, a2) | v7;
  sub_2CE0050((__int64)v10);
  return v8;
}
