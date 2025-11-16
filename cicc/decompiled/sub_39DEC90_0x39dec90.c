// Function: sub_39DEC90
// Address: 0x39dec90
//
void __fastcall sub_39DEC90(void **a1, __int64 a2)
{
  const char *v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // rdx
  _QWORD *v6; // rdx
  __int64 (*v7)(); // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r13
  unsigned __int64 v16; // r12
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // r14
  __int64 v19; // rbx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 *v35; // rbx
  unsigned __int64 *v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  unsigned __int64 v39; // r12
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 *v42; // rbx
  unsigned __int64 *v43; // r12
  __m128i v45[3]; // [rsp+50h] [rbp-310h] BYREF
  void *v46; // [rsp+80h] [rbp-2E0h] BYREF
  _BYTE *v47; // [rsp+88h] [rbp-2D8h]
  _BYTE *v48; // [rsp+90h] [rbp-2D0h]
  _BYTE *v49; // [rsp+98h] [rbp-2C8h]
  int v50; // [rsp+A0h] [rbp-2C0h]
  unsigned __int64 *v51; // [rsp+A8h] [rbp-2B8h]
  __int64 v52[4]; // [rsp+B0h] [rbp-2B0h] BYREF
  _BYTE *v53; // [rsp+D0h] [rbp-290h]
  __int64 v54; // [rsp+D8h] [rbp-288h]
  _BYTE v55[128]; // [rsp+E0h] [rbp-280h] BYREF
  const char *v56; // [rsp+160h] [rbp-200h] BYREF
  __int64 v57; // [rsp+168h] [rbp-1F8h]
  __int64 v58; // [rsp+170h] [rbp-1F0h]
  __int16 v59; // [rsp+178h] [rbp-1E8h]
  unsigned __int64 v60; // [rsp+180h] [rbp-1E0h]
  __int64 v61; // [rsp+188h] [rbp-1D8h]
  __int64 v62; // [rsp+190h] [rbp-1D0h]
  unsigned __int64 *v63; // [rsp+198h] [rbp-1C8h]
  unsigned __int64 *v64; // [rsp+1A0h] [rbp-1C0h]
  __int64 v65; // [rsp+1A8h] [rbp-1B8h]
  unsigned __int64 *v66; // [rsp+1B0h] [rbp-1B0h]
  unsigned __int64 *v67; // [rsp+1B8h] [rbp-1A8h]
  __int64 v68; // [rsp+1C8h] [rbp-198h]
  int v69; // [rsp+1D0h] [rbp-190h] BYREF
  __int64 v70; // [rsp+1D8h] [rbp-188h]
  __int64 v71; // [rsp+1E0h] [rbp-180h]
  __int16 v72; // [rsp+1E8h] [rbp-178h]
  _QWORD *v73; // [rsp+1F0h] [rbp-170h]
  __int64 v74; // [rsp+1F8h] [rbp-168h]
  _QWORD v75[4]; // [rsp+200h] [rbp-160h] BYREF
  int v76; // [rsp+220h] [rbp-140h]
  __int16 v77; // [rsp+224h] [rbp-13Ch]
  char v78; // [rsp+226h] [rbp-13Ah]
  int v79; // [rsp+228h] [rbp-138h]
  _QWORD *v80; // [rsp+230h] [rbp-130h]
  __int64 v81; // [rsp+238h] [rbp-128h]
  _QWORD v82[4]; // [rsp+240h] [rbp-120h] BYREF
  _QWORD *v83; // [rsp+260h] [rbp-100h]
  __int64 v84; // [rsp+268h] [rbp-F8h]
  _QWORD v85[4]; // [rsp+270h] [rbp-F0h] BYREF
  _QWORD *v86; // [rsp+290h] [rbp-D0h]
  _QWORD *v87; // [rsp+298h] [rbp-C8h]
  __int64 v88; // [rsp+2A0h] [rbp-C0h]
  _QWORD *v89; // [rsp+2A8h] [rbp-B8h]
  _QWORD *v90; // [rsp+2B0h] [rbp-B0h]
  __int64 v91; // [rsp+2B8h] [rbp-A8h]
  unsigned __int64 v92; // [rsp+2C0h] [rbp-A0h]
  __int64 v93; // [rsp+2C8h] [rbp-98h]
  __int64 v94; // [rsp+2D0h] [rbp-90h]
  int v95; // [rsp+2D8h] [rbp-88h] BYREF
  unsigned __int64 v96; // [rsp+2E0h] [rbp-80h]
  __int64 v97; // [rsp+2E8h] [rbp-78h]
  __int64 v98; // [rsp+2F0h] [rbp-70h]
  unsigned __int64 v99[2]; // [rsp+2F8h] [rbp-68h] BYREF
  _QWORD v100[11]; // [rsp+308h] [rbp-58h] BYREF

  sub_39DA270((__int64)a1, a2);
  v59 = 0;
  v73 = v75;
  v80 = v82;
  v77 = 0;
  v72 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v74 = 0;
  LOBYTE(v75[0]) = 0;
  v75[2] = 0;
  v75[3] = 0;
  v76 = -1;
  v78 = 0;
  v79 = 0;
  v81 = 0;
  LOBYTE(v82[0]) = 0;
  v82[2] = 0;
  v82[3] = 0;
  v83 = v85;
  v84 = 0;
  LOBYTE(v85[0]) = 0;
  v85[2] = 0;
  v85[3] = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 5;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99[0] = (unsigned __int64)v100;
  v99[1] = 0;
  LOBYTE(v100[0]) = 0;
  v100[2] = 0;
  v100[3] = 0;
  v2 = sub_1E0A440((__int64 *)a2);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = 0;
  v56 = v2;
  LODWORD(v2) = *(_DWORD *)(a2 + 340);
  v57 = v5;
  v6 = *(_QWORD **)(a2 + 352);
  LODWORD(v58) = (_DWORD)v2;
  BYTE4(v58) = *(_BYTE *)(a2 + 344);
  BYTE5(v58) = (*v6 & 0x20LL) != 0;
  BYTE6(v58) = (*v6 & 0x40LL) != 0;
  HIBYTE(v58) = *(_BYTE *)v6 >> 7;
  LOBYTE(v59) = (*v6 & 0x10LL) != 0;
  v7 = *(__int64 (**)())(*(_QWORD *)v3 + 112LL);
  if ( v7 != sub_1D00B10 )
    v4 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD *, _QWORD))v7)(v3, a2, v6, 0);
  sub_39D5BB0((__int64)a1, (__int64)&v56, *(_QWORD *)(a2 + 40), v4);
  sub_154BA10((__int64)v45, *(_QWORD *)(*(_QWORD *)a2 + 40LL), 1);
  sub_154C150((__int64)v45, *(_QWORD *)a2);
  sub_39D3240((__int64)a1, (__int64)v45, (__int64)&v69, *(_QWORD *)(a2 + 56));
  sub_39D7250((__int64)a1, &v56, a2, (__int64)v45);
  v8 = *(_QWORD *)(a2 + 64);
  if ( v8 )
    sub_39D8FF0((__int64)a1, (__int64)&v56, v8);
  v9 = *(_QWORD *)(a2 + 72);
  if ( v9 )
    sub_39D9B20(a1, v45, (__int64)&v95, v9);
  v10 = *(_QWORD *)(a2 + 328);
  v50 = 1;
  v49 = 0;
  v48 = 0;
  v46 = &unk_49EFBE0;
  v51 = v99;
  v47 = 0;
  if ( a2 + 320 != v10 )
  {
    while ( 1 )
    {
      v52[0] = (__int64)&v46;
      v52[1] = (__int64)v45;
      v52[2] = (__int64)(a1 + 1);
      v53 = v55;
      v52[3] = (__int64)(a1 + 5);
      v54 = 0x800000000LL;
      sub_39D4670(v52, v10);
      if ( v53 != v55 )
        _libc_free((unsigned __int64)v53);
      v10 = *(_QWORD *)(v10 + 8);
      if ( a2 + 320 == v10 )
        break;
      if ( v48 == v49 )
        sub_16E7EE0((__int64)&v46, "\n", 1u);
      else
        *v49++ = 10;
    }
    if ( v47 != v49 )
      sub_16E7BA0((__int64 *)&v46);
  }
  sub_16E4AB0((__int64)v52, (__int64)*a1, 0, 70);
  if ( !byte_50576C0 )
    v55[48] = 1;
  nullsub_622();
  if ( (unsigned __int8)sub_16E4B20() )
  {
    sub_16E3D10((__int64)v52, 0, v11, v12, v13, v14);
    sub_39DCE30((__int64)v52, (__int64)&v56);
    sub_16E3410((__int64)v52);
    nullsub_628();
  }
  sub_16E4BA0((__int64)v52);
  sub_16E3E40(v52);
  sub_16E7BC0((__int64 *)&v46);
  sub_154BA40(v45[0].m128i_i64);
  if ( (_QWORD *)v99[0] != v100 )
    j_j___libc_free_0(v99[0]);
  v15 = v97;
  v16 = v96;
  if ( v97 != v96 )
  {
    do
    {
      v17 = *(unsigned __int64 **)(v16 + 32);
      v18 = *(unsigned __int64 **)(v16 + 24);
      if ( v17 != v18 )
      {
        do
        {
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            j_j___libc_free_0(*v18);
          v18 += 6;
        }
        while ( v17 != v18 );
        v18 = *(unsigned __int64 **)(v16 + 24);
      }
      if ( v18 )
        j_j___libc_free_0((unsigned __int64)v18);
      v16 += 48LL;
    }
    while ( v15 != v16 );
    v16 = v96;
  }
  if ( v16 )
    j_j___libc_free_0(v16);
  v19 = v93;
  v20 = v92;
  if ( v93 != v92 )
  {
    do
    {
      v21 = *(_QWORD *)(v20 + 24);
      if ( v21 != v20 + 40 )
        j_j___libc_free_0(v21);
      v20 += 80LL;
    }
    while ( v19 != v20 );
    v20 = v92;
  }
  if ( v20 )
    j_j___libc_free_0(v20);
  v22 = v90;
  v23 = v89;
  if ( v90 != v89 )
  {
    do
    {
      v24 = v23[34];
      if ( (_QWORD *)v24 != v23 + 36 )
        j_j___libc_free_0(v24);
      v25 = v23[28];
      if ( (_QWORD *)v25 != v23 + 30 )
        j_j___libc_free_0(v25);
      v26 = v23[22];
      if ( (_QWORD *)v26 != v23 + 24 )
        j_j___libc_free_0(v26);
      v27 = v23[13];
      if ( (_QWORD *)v27 != v23 + 15 )
        j_j___libc_free_0(v27);
      v28 = v23[3];
      if ( (_QWORD *)v28 != v23 + 5 )
        j_j___libc_free_0(v28);
      v23 += 40;
    }
    while ( v22 != v23 );
    v23 = v89;
  }
  if ( v23 )
    j_j___libc_free_0((unsigned __int64)v23);
  v29 = v87;
  v30 = v86;
  if ( v87 != v86 )
  {
    do
    {
      v31 = v30[26];
      if ( (_QWORD *)v31 != v30 + 28 )
        j_j___libc_free_0(v31);
      v32 = v30[20];
      if ( (_QWORD *)v32 != v30 + 22 )
        j_j___libc_free_0(v32);
      v33 = v30[14];
      if ( (_QWORD *)v33 != v30 + 16 )
        j_j___libc_free_0(v33);
      v34 = v30[7];
      if ( (_QWORD *)v34 != v30 + 9 )
        j_j___libc_free_0(v34);
      v30 += 32;
    }
    while ( v29 != v30 );
    v30 = v86;
  }
  if ( v30 )
    j_j___libc_free_0((unsigned __int64)v30);
  if ( v83 != v85 )
    j_j___libc_free_0((unsigned __int64)v83);
  if ( v80 != v82 )
    j_j___libc_free_0((unsigned __int64)v80);
  if ( v73 != v75 )
    j_j___libc_free_0((unsigned __int64)v73);
  if ( (_BYTE)v68 )
  {
    v42 = v67;
    v43 = v66;
    if ( v67 != v66 )
    {
      do
      {
        if ( (unsigned __int64 *)*v43 != v43 + 2 )
          j_j___libc_free_0(*v43);
        v43 += 6;
      }
      while ( v42 != v43 );
      v43 = v66;
    }
    if ( v43 )
      j_j___libc_free_0((unsigned __int64)v43);
  }
  v35 = v64;
  v36 = v63;
  if ( v64 != v63 )
  {
    do
    {
      v37 = v36[6];
      if ( (unsigned __int64 *)v37 != v36 + 8 )
        j_j___libc_free_0(v37);
      if ( (unsigned __int64 *)*v36 != v36 + 2 )
        j_j___libc_free_0(*v36);
      v36 += 12;
    }
    while ( v35 != v36 );
    v36 = v63;
  }
  if ( v36 )
    j_j___libc_free_0((unsigned __int64)v36);
  v38 = v61;
  v39 = v60;
  if ( v61 != v60 )
  {
    do
    {
      v40 = *(_QWORD *)(v39 + 72);
      if ( v40 != v39 + 88 )
        j_j___libc_free_0(v40);
      v41 = *(_QWORD *)(v39 + 24);
      if ( v41 != v39 + 40 )
        j_j___libc_free_0(v41);
      v39 += 120LL;
    }
    while ( v38 != v39 );
    v39 = v60;
  }
  if ( v39 )
    j_j___libc_free_0(v39);
}
