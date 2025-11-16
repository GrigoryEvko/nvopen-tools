// Function: sub_2C883F0
// Address: 0x2c883f0
//
__int64 __fastcall sub_2C883F0(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // bl
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rdi
  void *v10; // rsi
  bool v12; // al
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rdi
  __int64 v17; // [rsp+20h] [rbp-2F0h] BYREF
  void **v18; // [rsp+28h] [rbp-2E8h]
  __int64 v19; // [rsp+30h] [rbp-2E0h]
  int v20; // [rsp+38h] [rbp-2D8h]
  char v21; // [rsp+3Ch] [rbp-2D4h]
  void *v22; // [rsp+40h] [rbp-2D0h] BYREF
  unsigned __int64 v23; // [rsp+48h] [rbp-2C8h]
  void **v24; // [rsp+50h] [rbp-2C0h] BYREF
  void **v25; // [rsp+58h] [rbp-2B8h]
  __int64 v26; // [rsp+60h] [rbp-2B0h]
  int v27; // [rsp+68h] [rbp-2A8h]
  char v28; // [rsp+6Ch] [rbp-2A4h]
  int v29; // [rsp+70h] [rbp-2A0h] BYREF
  unsigned __int64 v30; // [rsp+78h] [rbp-298h]
  int *v31; // [rsp+80h] [rbp-290h]
  int *v32; // [rsp+88h] [rbp-288h]
  __int64 v33; // [rsp+90h] [rbp-280h]
  int v34; // [rsp+A0h] [rbp-270h] BYREF
  unsigned __int64 v35; // [rsp+A8h] [rbp-268h]
  int *v36; // [rsp+B0h] [rbp-260h]
  int *v37; // [rsp+B8h] [rbp-258h]
  __int64 v38; // [rsp+C0h] [rbp-250h]
  int v39; // [rsp+D0h] [rbp-240h] BYREF
  unsigned __int64 v40; // [rsp+D8h] [rbp-238h]
  int *v41; // [rsp+E0h] [rbp-230h]
  int *v42; // [rsp+E8h] [rbp-228h]
  __int64 v43; // [rsp+F0h] [rbp-220h]
  int v44; // [rsp+100h] [rbp-210h] BYREF
  unsigned __int64 v45; // [rsp+108h] [rbp-208h]
  int *v46; // [rsp+110h] [rbp-200h]
  int *v47; // [rsp+118h] [rbp-1F8h]
  __int64 v48; // [rsp+120h] [rbp-1F0h]
  int v49; // [rsp+130h] [rbp-1E0h] BYREF
  unsigned __int64 v50; // [rsp+138h] [rbp-1D8h]
  int *v51; // [rsp+140h] [rbp-1D0h]
  int *v52; // [rsp+148h] [rbp-1C8h]
  __int64 v53; // [rsp+150h] [rbp-1C0h]
  int v54; // [rsp+160h] [rbp-1B0h] BYREF
  unsigned __int64 v55; // [rsp+168h] [rbp-1A8h]
  int *v56; // [rsp+170h] [rbp-1A0h]
  int *v57; // [rsp+178h] [rbp-198h]
  __int64 v58; // [rsp+180h] [rbp-190h]
  int v59; // [rsp+190h] [rbp-180h] BYREF
  unsigned __int64 v60; // [rsp+198h] [rbp-178h]
  int *v61; // [rsp+1A0h] [rbp-170h]
  int *v62; // [rsp+1A8h] [rbp-168h]
  __int64 v63; // [rsp+1B0h] [rbp-160h]
  int v64; // [rsp+1C0h] [rbp-150h] BYREF
  unsigned __int64 v65; // [rsp+1C8h] [rbp-148h]
  int *v66; // [rsp+1D0h] [rbp-140h]
  int *v67; // [rsp+1D8h] [rbp-138h]
  __int64 v68; // [rsp+1E0h] [rbp-130h]
  int v69; // [rsp+1F0h] [rbp-120h] BYREF
  unsigned __int64 v70; // [rsp+1F8h] [rbp-118h]
  int *v71; // [rsp+200h] [rbp-110h]
  int *v72; // [rsp+208h] [rbp-108h]
  __int64 v73; // [rsp+210h] [rbp-100h]
  int v74; // [rsp+220h] [rbp-F0h] BYREF
  unsigned __int64 v75; // [rsp+228h] [rbp-E8h]
  int *v76; // [rsp+230h] [rbp-E0h]
  int *v77; // [rsp+238h] [rbp-D8h]
  __int64 v78; // [rsp+240h] [rbp-D0h]
  int v79; // [rsp+250h] [rbp-C0h] BYREF
  unsigned __int64 v80; // [rsp+258h] [rbp-B8h]
  int *v81; // [rsp+260h] [rbp-B0h]
  int *v82; // [rsp+268h] [rbp-A8h]
  __int64 v83; // [rsp+270h] [rbp-A0h]
  int v84; // [rsp+280h] [rbp-90h] BYREF
  unsigned __int64 v85; // [rsp+288h] [rbp-88h]
  int *v86; // [rsp+290h] [rbp-80h]
  int *v87; // [rsp+298h] [rbp-78h]
  __int64 v88; // [rsp+2A0h] [rbp-70h]
  int v89; // [rsp+2B0h] [rbp-60h] BYREF
  unsigned __int64 v90; // [rsp+2B8h] [rbp-58h]
  int *v91; // [rsp+2C0h] [rbp-50h]
  int *v92; // [rsp+2C8h] [rbp-48h]
  __int64 v93; // [rsp+2D0h] [rbp-40h]

  v18 = 0;
  v17 = (__int64)sub_CB72A0();
  v36 = &v34;
  v37 = &v34;
  v41 = &v39;
  v42 = &v39;
  v46 = &v44;
  v47 = &v44;
  v51 = &v49;
  v19 = 0;
  LODWORD(v22) = 0;
  v23 = 0;
  v24 = &v22;
  v25 = &v22;
  v26 = 0;
  v29 = 0;
  v30 = 0;
  v31 = &v29;
  v32 = &v29;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v52 = &v49;
  v56 = &v54;
  v57 = &v54;
  v61 = &v59;
  v62 = &v59;
  v66 = &v64;
  v67 = &v64;
  v71 = &v69;
  v72 = &v69;
  v76 = &v74;
  v77 = &v74;
  v81 = &v79;
  v82 = &v79;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v5 = qword_5011228;
  v86 = &v84;
  v87 = &v84;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = &v89;
  v92 = &v89;
  v93 = 0;
  if ( (_BYTE)qword_5011228 )
  {
    v12 = sub_2C84BA0(&v17, a3);
    v13 = v90;
    v5 = v12;
    while ( v13 )
    {
      sub_2C83D50(*(_QWORD *)(v13 + 24));
      v14 = v13;
      v13 = *(_QWORD *)(v13 + 16);
      j_j___libc_free_0(v14);
    }
    v15 = v85;
    while ( v15 )
    {
      sub_2C83D50(*(_QWORD *)(v15 + 24));
      v16 = v15;
      v15 = *(_QWORD *)(v15 + 16);
      j_j___libc_free_0(v16);
    }
  }
  v6 = v80;
  while ( v6 )
  {
    sub_2C83D50(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
  v8 = v75;
  while ( v8 )
  {
    sub_2C83D50(*(_QWORD *)(v8 + 24));
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v9);
  }
  sub_2C84080(v70);
  sub_2C84080(v65);
  sub_2C84080(v60);
  sub_2C84080(v55);
  sub_2C84080(v50);
  sub_2C84080(v45);
  sub_2C84080(v40);
  sub_2C84080(v35);
  sub_2C84080(v30);
  sub_2C84080(v23);
  v10 = (void *)(a1 + 32);
  if ( v5 )
  {
    v22 = &unk_4F82408;
    v18 = &v22;
    v19 = 0x100000002LL;
    v20 = 0;
    v21 = 1;
    v24 = 0;
    v25 = (void **)&v29;
    v26 = 2;
    v27 = 0;
    v28 = 1;
    v17 = 1;
    sub_C8CF70(a1, v10, 2, (__int64)&v22, (__int64)&v17);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v29, (__int64)&v24);
    if ( !v28 )
      _libc_free((unsigned __int64)v25);
    if ( !v21 )
      _libc_free((unsigned __int64)v18);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v10;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
