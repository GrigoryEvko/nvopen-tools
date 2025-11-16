// Function: sub_35A6DE0
// Address: 0x35a6de0
//
__int64 __fastcall sub_35A6DE0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v5; // rsi
  _QWORD *(__fastcall *v6)(_QWORD *); // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 *v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 *v36; // r13
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // r15
  __int64 v44; // r13
  __int64 (*v45)(); // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 (*v50)(); // rax
  void (__fastcall *v51)(__int64, __int64 *, __int64 *, __int64 *, _BYTE *, _QWORD, _BYTE **); // rax
  __int64 v52; // r11
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rsi
  __int64 v57; // rdi
  __int64 v58; // rdx
  __int64 v59; // rsi
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rax
  __int64 v88; // r9
  __int64 v89; // r8
  int v90; // edx
  __int64 v91; // rsi
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // r8
  __int64 v98; // r9
  __int64 v99; // r13
  _BYTE *v100; // r12
  __int64 v101; // rsi
  __int64 v102; // rdi
  __int64 v103; // r13
  _BYTE *v104; // r12
  __int64 v105; // rsi
  __int64 v106; // rdi
  __int64 *v107; // r13
  __int64 *v108; // r12
  __int64 v109; // rsi
  __int64 v110; // rdi
  __int64 v112; // [rsp+8h] [rbp-188h]
  __int64 v113; // [rsp+20h] [rbp-170h]
  __int64 v114; // [rsp+28h] [rbp-168h]
  __int64 *v115; // [rsp+30h] [rbp-160h] BYREF
  __int64 v116; // [rsp+38h] [rbp-158h]
  __int64 v117; // [rsp+40h] [rbp-150h]
  unsigned int v118; // [rsp+48h] [rbp-148h]
  __int64 *v119; // [rsp+50h] [rbp-140h] BYREF
  __int64 v120; // [rsp+58h] [rbp-138h]
  _BYTE v121[32]; // [rsp+60h] [rbp-130h] BYREF
  _BYTE *v122; // [rsp+80h] [rbp-110h] BYREF
  __int64 v123; // [rsp+88h] [rbp-108h]
  _BYTE v124[32]; // [rsp+90h] [rbp-100h] BYREF
  _BYTE *v125; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v126; // [rsp+B8h] [rbp-D8h]
  _BYTE v127[208]; // [rsp+C0h] [rbp-D0h] BYREF

  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_QWORD *(__fastcall **)(_QWORD *))(*(_QWORD *)v5 + 376LL);
  v7 = 0;
  if ( v6 != sub_2FDC520 )
  {
    ((void (__fastcall *)(_BYTE **, __int64, _QWORD))v6)(&v125, v5, *(_QWORD *)(a1 + 48));
    v7 = (__int64)v125;
  }
  v125 = 0;
  v8 = *(_QWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 120) = v7;
  if ( v8 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
    if ( v125 )
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v125 + 8LL))(v125);
  }
  sub_35A3A90(a1);
  v9 = *(_QWORD *)(a1 + 48);
  LOBYTE(v126) = 0;
  v10 = sub_2E7AAE0(*(_QWORD *)(a1 + 8), *(_QWORD *)(v9 + 16), (__int64)v125, 0);
  LOBYTE(v126) = 0;
  v11 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 72) = v10;
  v12 = sub_2E7AAE0(v11, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), (__int64)v125, v126);
  LOBYTE(v126) = 0;
  v13 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 80) = v12;
  v14 = sub_2E7AAE0(v13, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), (__int64)v125, v126);
  LOBYTE(v126) = 0;
  v15 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 88) = v14;
  v16 = sub_2E7AAE0(v15, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), (__int64)v125, v126);
  LOBYTE(v126) = 0;
  v17 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 96) = v16;
  v18 = sub_2E7AAE0(v17, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), (__int64)v125, v126);
  v19 = *(_QWORD *)(a1 + 72);
  v20 = *(__int64 **)(a1 + 48);
  *(_QWORD *)(a1 + 104) = v18;
  sub_2E33BD0(*(_QWORD *)(a1 + 8) + 320LL, v19);
  v21 = *v20;
  v22 = *(_QWORD *)v19;
  *(_QWORD *)(v19 + 8) = v20;
  v21 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v19 = v21 | v22 & 7;
  *(_QWORD *)(v21 + 8) = v19;
  *v20 = *v20 & 7 | v19;
  v23 = *(_QWORD *)(a1 + 80);
  v24 = *(__int64 **)(a1 + 48);
  sub_2E33BD0(*(_QWORD *)(a1 + 8) + 320LL, v23);
  v25 = *v24;
  v26 = *(_QWORD *)v23;
  *(_QWORD *)(v23 + 8) = v24;
  v25 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v23 = v25 | v26 & 7;
  *(_QWORD *)(v25 + 8) = v23;
  *v24 = *v24 & 7 | v23;
  v27 = *(_QWORD *)(a1 + 88);
  v28 = *(__int64 **)(a1 + 48);
  sub_2E33BD0(*(_QWORD *)(a1 + 8) + 320LL, v27);
  v29 = *v28;
  v30 = *(_QWORD *)v27;
  *(_QWORD *)(v27 + 8) = v28;
  v29 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v27 = v29 | v30 & 7;
  *(_QWORD *)(v29 + 8) = v27;
  *v28 = *v28 & 7 | v27;
  v31 = *(_QWORD *)(a1 + 96);
  v32 = *(__int64 **)(a1 + 48);
  sub_2E33BD0(*(_QWORD *)(a1 + 8) + 320LL, v31);
  v33 = *v32;
  v34 = *(_QWORD *)v31;
  *(_QWORD *)(v31 + 8) = v32;
  v33 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v31 = v33 | v34 & 7;
  *(_QWORD *)(v33 + 8) = v31;
  *v32 = *v32 & 7 | v31;
  v35 = *(_QWORD *)(a1 + 104);
  v36 = *(__int64 **)(a1 + 48);
  sub_2E33BD0(*(_QWORD *)(a1 + 8) + 320LL, v35);
  v40 = *v36;
  v41 = *(_QWORD *)v35;
  *(_QWORD *)(v35 + 8) = v36;
  v42 = v40 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v35 = v42 | v41 & 7;
  *(_QWORD *)(v42 + 8) = v35;
  *v36 = *v36 & 7 | v35;
  v114 = *(_QWORD *)(a1 + 64);
  if ( *(_DWORD *)(v114 + 72) == 1 )
  {
    v47 = *(_QWORD *)(a1 + 64);
    goto LABEL_17;
  }
  v43 = *(__int64 **)(a1 + 48);
  v44 = v43[4];
  v45 = *(__int64 (**)())(**(_QWORD **)(v44 + 16) + 128LL);
  if ( v45 == sub_2DAC790 )
  {
    LOBYTE(v126) = 0;
    v1 = sub_2E7AAE0(v44, v43[2], (__int64)v125, 0);
    sub_2E33BD0(v44 + 320, v1);
    v2 = *v43;
    v3 = *(_QWORD *)v1;
    *(_QWORD *)(v1 + 8) = v43;
    v2 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v1 = v2 | v3 & 7;
    *(_QWORD *)(v2 + 8) = v1;
    *v43 = *v43 & 7 | v1;
    v125 = v127;
    v115 = 0;
    v119 = 0;
    v126 = 0x400000000LL;
    BUG();
  }
  v46 = v45();
  LOBYTE(v126) = 0;
  v113 = v46;
  v47 = sub_2E7AAE0(v44, v43[2], (__int64)v125, 0);
  sub_2E33BD0(v44 + 320, v47);
  v48 = *v43;
  v49 = *(_QWORD *)v47;
  *(_QWORD *)(v47 + 8) = v43;
  v48 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v47 = v48 | v49 & 7;
  *(_QWORD *)(v48 + 8) = v47;
  *v43 = v47 | *v43 & 7;
  v125 = v127;
  v115 = 0;
  v119 = 0;
  v126 = 0x400000000LL;
  v50 = *(__int64 (**)())(*(_QWORD *)v113 + 344LL);
  if ( v50 == sub_2DB1AE0 )
    goto LABEL_36;
  ((void (__fastcall *)(__int64, __int64 *, __int64 **, __int64 **, _BYTE **, _QWORD))v50)(
    v113,
    v43,
    &v115,
    &v119,
    &v125,
    0);
  if ( v43 != v115 )
  {
    if ( v43 == v119 )
    {
      v115 = (__int64 *)v47;
      goto LABEL_11;
    }
LABEL_36:
    BUG();
  }
  v119 = (__int64 *)v47;
LABEL_11:
  (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)v113 + 360LL))(v113, v43, 0);
  v51 = *(void (__fastcall **)(__int64, __int64 *, __int64 *, __int64 *, _BYTE *, _QWORD, _BYTE **))(*(_QWORD *)v113 + 368LL);
  v122 = 0;
  v51(v113, v43, v115, v119, v125, (unsigned int)v126, &v122);
  v52 = v113;
  if ( v122 )
  {
    sub_B91220((__int64)&v122, (__int64)v122);
    v52 = v113;
  }
  v112 = v52;
  sub_2E33690((__int64)v43, v114, v47);
  v122 = 0;
  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)v112 + 368LL))(
    v112,
    v47,
    v114,
    0,
    0,
    0,
    &v122,
    0);
  if ( v122 )
    sub_B91220((__int64)&v122, (__int64)v122);
  sub_2E33F80(v47, v114, -1, v53, v54, v55);
  sub_2E32770(v114, (__int64)v43, v47);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
LABEL_17:
  v56 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 112) = v47;
  sub_2E34140(*(_QWORD *)(a1 + 104), v56, v42, v37, v38, v39);
  v57 = *(_QWORD *)(a1 + 32);
  v58 = *(_QWORD *)(a1 + 48);
  v59 = *(_QWORD *)(a1 + 104);
  v125 = 0;
  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)v57 + 368LL))(
    v57,
    v59,
    v58,
    0,
    0,
    0,
    &v125,
    0);
  sub_9C6650(&v125);
  sub_2E33F80(*(_QWORD *)(a1 + 56), *(_QWORD *)(a1 + 72), -1, v60, v61, v62);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 32) + 360LL))(
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 56),
    0);
  v63 = *(_QWORD *)(a1 + 32);
  v64 = *(_QWORD *)(a1 + 72);
  v65 = *(_QWORD *)(a1 + 56);
  v125 = 0;
  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _BYTE **, _QWORD))(*(_QWORD *)v63 + 368LL))(
    v63,
    v65,
    v64,
    0,
    0,
    0,
    &v125,
    0);
  sub_9C6650(&v125);
  sub_2E33F80(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), -1, v66, v67, v68);
  sub_2E33F80(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 104), -1, v69, v70, v71);
  sub_2E33F80(*(_QWORD *)(a1 + 80), *(_QWORD *)(a1 + 88), -1, v72, v73, v74);
  sub_2E33F80(*(_QWORD *)(a1 + 88), *(_QWORD *)(a1 + 88), -1, v75, v76, v77);
  sub_2E33F80(*(_QWORD *)(a1 + 88), *(_QWORD *)(a1 + 96), -1, v78, v79, v80);
  sub_2E33F80(*(_QWORD *)(a1 + 96), *(_QWORD *)(a1 + 104), -1, v81, v82, v83);
  sub_2E33F80(*(_QWORD *)(a1 + 96), *(_QWORD *)(a1 + 112), -1, v84, v85, v86);
  v87 = *(_QWORD *)a1;
  v88 = *(_QWORD *)(a1 + 104);
  v89 = *(_QWORD *)(a1 + 80);
  v90 = *(_DWORD *)(a1 + 128);
  v115 = 0;
  v91 = *(_QWORD *)(a1 + 72);
  v116 = 0;
  v117 = 0;
  v118 = 0;
  sub_359A2D0(a1, v91, *(_DWORD *)(v87 + 96) + v90 - 2, (__int64)&v115, v89, v88);
  v119 = (__int64 *)v121;
  v120 = 0x100000000LL;
  v123 = 0x100000000LL;
  v126 = 0x100000000LL;
  v125 = v127;
  v122 = v124;
  sub_35A67E0((__int64 *)a1, (__int64)&v119, (__int64)v127, v92, v93, v94);
  sub_35A5FD0((int *)a1, &v119, (__int64)&v122, (__int64)&v115, v95, v96);
  sub_35A59B0((__int64 *)a1, (__int64)&v122, (__int64)&v125, (__int64)&v115, v97, v98);
  v99 = (__int64)v125;
  v100 = &v125[32 * (unsigned int)v126];
  if ( v125 != v100 )
  {
    do
    {
      v101 = *((unsigned int *)v100 - 2);
      v102 = *((_QWORD *)v100 - 3);
      v100 -= 32;
      sub_C7D6A0(v102, 8 * v101, 4);
    }
    while ( (_BYTE *)v99 != v100 );
    v100 = v125;
  }
  if ( v100 != v127 )
    _libc_free((unsigned __int64)v100);
  v103 = (__int64)v122;
  v104 = &v122[32 * (unsigned int)v123];
  if ( v122 != v104 )
  {
    do
    {
      v105 = *((unsigned int *)v104 - 2);
      v106 = *((_QWORD *)v104 - 3);
      v104 -= 32;
      sub_C7D6A0(v106, 8 * v105, 4);
    }
    while ( (_BYTE *)v103 != v104 );
    v104 = v122;
  }
  if ( v104 != v124 )
    _libc_free((unsigned __int64)v104);
  v107 = v119;
  v108 = &v119[4 * (unsigned int)v120];
  if ( v119 != v108 )
  {
    do
    {
      v109 = *((unsigned int *)v108 - 2);
      v110 = *(v108 - 3);
      v108 -= 4;
      sub_C7D6A0(v110, 8 * v109, 4);
    }
    while ( v107 != v108 );
    v108 = v119;
  }
  if ( v108 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v108);
  return sub_C7D6A0(v116, 16LL * v118, 8);
}
