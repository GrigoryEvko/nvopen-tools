// Function: sub_20C9DC0
// Address: 0x20c9dc0
//
__int64 __fastcall sub_20C9DC0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 (__fastcall *a5)(__int64, __int64 *, __int64),
        __int64 a6,
        void (__fastcall *a7)(__int64, __int64 *, __int64, __int64, __int64, _QWORD, unsigned __int8 **, __int64 *),
        __int64 a8)
{
  __int64 v10; // r13
  _QWORD *v11; // rdi
  __int64 *v12; // rsi
  __int64 v13; // rbx
  _QWORD *v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // rdi
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  unsigned int v26; // eax
  _QWORD *v27; // rax
  _QWORD *v28; // r13
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rdx
  unsigned __int8 *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rdi
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int8 *v46; // rsi
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  unsigned int v59; // eax
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rbx
  int v63; // eax
  __int64 v64; // rax
  int v65; // edx
  __int64 v66; // rdx
  _QWORD *v67; // rax
  __int64 v68; // rcx
  unsigned __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // r15
  _QWORD *v75; // rax
  _QWORD *v76; // rbx
  __int64 v77; // rdi
  unsigned __int64 *v78; // r13
  __int64 v79; // rax
  unsigned __int64 v80; // rcx
  __int64 v81; // rsi
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rsi
  unsigned __int8 *v87; // rsi
  __int64 v89; // [rsp-10h] [rbp-F0h]
  __int64 v90; // [rsp-8h] [rbp-E8h]
  _QWORD *v95; // [rsp+38h] [rbp-A8h]
  __int64 v96; // [rsp+48h] [rbp-98h]
  __int64 v97; // [rsp+50h] [rbp-90h]
  __int64 v98; // [rsp+58h] [rbp-88h]
  unsigned __int64 *v99; // [rsp+58h] [rbp-88h]
  __int64 v100; // [rsp+58h] [rbp-88h]
  __int64 v101; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int8 *v102; // [rsp+68h] [rbp-78h] BYREF
  __int64 v103[2]; // [rsp+70h] [rbp-70h] BYREF
  char v104; // [rsp+80h] [rbp-60h]
  char v105; // [rsp+81h] [rbp-5Fh]
  __int64 v106[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v107; // [rsp+A0h] [rbp-40h]

  v10 = a1[3];
  v11 = (_QWORD *)a1[1];
  v12 = (__int64 *)a1[2];
  v13 = v11[7];
  v95 = v11;
  v106[0] = (__int64)"atomicrmw.end";
  v107 = 259;
  v97 = sub_157FBF0(v11, v12, (__int64)v106);
  v106[0] = (__int64)"atomicrmw.start";
  v107 = 259;
  v14 = (_QWORD *)sub_22077B0(64);
  v96 = (__int64)v14;
  if ( v14 )
    sub_157FB60(v14, v10, (__int64)v106, v13, v97);
  v15 = v11 + 5;
  v98 = v11[5];
  v16 = (_QWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v98 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v16 = 0;
  sub_15F20C0(v16);
  a1[2] = (__int64)v15;
  a1[1] = (__int64)v95;
  v107 = 257;
  v17 = sub_1648A60(64, 1u);
  v18 = v17;
  if ( v17 )
    sub_15F9210((__int64)v17, a2, a3, 0, 0, 0);
  v19 = a1[1];
  if ( v19 )
  {
    v20 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v19 + 40, (__int64)v18);
    v21 = v18[3];
    v22 = *v20;
    v18[4] = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v18 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780((__int64)v18, v106);
  v23 = *a1;
  if ( *a1 )
  {
    v103[0] = *a1;
    sub_1623A60((__int64)v103, v23, 2);
    v24 = v18[6];
    if ( v24 )
      sub_161E7C0((__int64)(v18 + 6), v24);
    v25 = (unsigned __int8 *)v103[0];
    v18[6] = v103[0];
    if ( v25 )
      sub_1623210((__int64)v103, v25, (__int64)(v18 + 6));
  }
  v26 = sub_1643030(a2);
  sub_15F8F50((__int64)v18, v26 >> 3);
  v107 = 257;
  v27 = sub_1648A60(56, 1u);
  v28 = v27;
  if ( v27 )
    sub_15F8320((__int64)v27, v96, 0);
  v29 = a1[1];
  if ( v29 )
  {
    v99 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v29 + 40, (__int64)v28);
    v30 = *v99;
    v31 = v28[3] & 7LL;
    v28[4] = v99;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    v28[3] = v30 | v31;
    *(_QWORD *)(v30 + 8) = v28 + 3;
    *v99 = *v99 & 7 | (unsigned __int64)(v28 + 3);
  }
  sub_164B780((__int64)v28, v106);
  v32 = *a1;
  if ( *a1 )
  {
    v103[0] = *a1;
    sub_1623A60((__int64)v103, v32, 2);
    v33 = v28[6];
    v34 = (__int64)(v28 + 6);
    if ( v33 )
    {
      sub_161E7C0((__int64)(v28 + 6), v33);
      v34 = (__int64)(v28 + 6);
    }
    v35 = (unsigned __int8 *)v103[0];
    v28[6] = v103[0];
    if ( v35 )
      sub_1623210((__int64)v103, v35, v34);
  }
  v105 = 1;
  v104 = 3;
  a1[1] = v96;
  a1[2] = v96 + 40;
  v103[0] = (__int64)"loaded";
  v107 = 257;
  v36 = sub_1648B60(64);
  v37 = v36;
  if ( v36 )
  {
    v100 = v36;
    sub_15F1EA0(v36, a2, 53, 0, 0, 0);
    *(_DWORD *)(v37 + 56) = 2;
    sub_164B780(v37, v106);
    sub_1648880(v37, *(_DWORD *)(v37 + 56), 1);
  }
  else
  {
    v100 = 0;
  }
  v38 = a1[1];
  if ( v38 )
  {
    v39 = (__int64 *)a1[2];
    sub_157E9D0(v38 + 40, v37);
    v40 = *(_QWORD *)(v37 + 24);
    v41 = *v39;
    *(_QWORD *)(v37 + 32) = v39;
    v41 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v37 + 24) = v41 | v40 & 7;
    *(_QWORD *)(v41 + 8) = v37 + 24;
    *v39 = *v39 & 7 | (v37 + 24);
  }
  sub_164B780(v100, v103);
  v46 = (unsigned __int8 *)*a1;
  if ( *a1 )
  {
    v102 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v102, (__int64)v46, 2);
    v47 = *(_QWORD *)(v37 + 48);
    v42 = v37 + 48;
    if ( v47 )
    {
      sub_161E7C0(v37 + 48, v47);
      v42 = v37 + 48;
    }
    v46 = v102;
    *(_QWORD *)(v37 + 48) = v102;
    if ( v46 )
      sub_1623210((__int64)&v102, v46, v42);
  }
  v48 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  if ( v48 == *(_DWORD *)(v37 + 56) )
  {
    sub_15F55D0(v37, (__int64)v46, v42, v43, v44, v45);
    v48 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  }
  v49 = (v48 + 1) & 0xFFFFFFF;
  v50 = v49 | *(_DWORD *)(v37 + 20) & 0xF0000000;
  *(_DWORD *)(v37 + 20) = v50;
  if ( (v50 & 0x40000000) != 0 )
    v51 = *(_QWORD *)(v37 - 8);
  else
    v51 = v100 - 24 * v49;
  v52 = (_QWORD *)(v51 + 24LL * (unsigned int)(v49 - 1));
  if ( *v52 )
  {
    v53 = v52[1];
    v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v54 = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = *(_QWORD *)(v53 + 16) & 3LL | v54;
  }
  *v52 = v18;
  if ( v18 )
  {
    v55 = v18[1];
    v52[1] = v55;
    if ( v55 )
      *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
    v52[2] = (unsigned __int64)(v18 + 1) | v52[2] & 3LL;
    v18[1] = v52;
  }
  v56 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
    v57 = *(_QWORD *)(v37 - 8);
  else
    v57 = v100 - 24 * v56;
  *(_QWORD *)(v57 + 8LL * (unsigned int)(v56 - 1) + 24LL * *(unsigned int *)(v37 + 56) + 8) = v95;
  v58 = a5(a6, a1, v37);
  v59 = 2;
  v101 = 0;
  v102 = 0;
  if ( a4 != 1 )
    v59 = a4;
  a7(a8, a1, a3, v37, v58, v59, &v102, &v101);
  v62 = v101;
  v63 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  if ( v63 == *(_DWORD *)(v37 + 56) )
  {
    sub_15F55D0(v37, (__int64)a1, v89, v90, v60, v61);
    v63 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  }
  v64 = (v63 + 1) & 0xFFFFFFF;
  v65 = v64 | *(_DWORD *)(v37 + 20) & 0xF0000000;
  *(_DWORD *)(v37 + 20) = v65;
  if ( (v65 & 0x40000000) != 0 )
    v66 = *(_QWORD *)(v37 - 8);
  else
    v66 = v100 - 24 * v64;
  v67 = (_QWORD *)(v66 + 24LL * (unsigned int)(v64 - 1));
  if ( *v67 )
  {
    v68 = v67[1];
    v69 = v67[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v69 = v68;
    if ( v68 )
      *(_QWORD *)(v68 + 16) = *(_QWORD *)(v68 + 16) & 3LL | v69;
  }
  *v67 = v62;
  if ( v62 )
  {
    v70 = *(_QWORD *)(v62 + 8);
    v67[1] = v70;
    if ( v70 )
      *(_QWORD *)(v70 + 16) = (unsigned __int64)(v67 + 1) | *(_QWORD *)(v70 + 16) & 3LL;
    v67[2] = (v62 + 8) | v67[2] & 3LL;
    *(_QWORD *)(v62 + 8) = v67;
  }
  v71 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  v72 = (unsigned int)(v71 - 1);
  if ( (*(_BYTE *)(v37 + 23) & 0x40) != 0 )
    v73 = *(_QWORD *)(v37 - 8);
  else
    v73 = v100 - 24 * v71;
  *(_QWORD *)(v73 + 8 * v72 + 24LL * *(unsigned int *)(v37 + 56) + 8) = v96;
  v74 = (__int64)v102;
  v107 = 257;
  v75 = sub_1648A60(56, 3u);
  v76 = v75;
  if ( v75 )
    sub_15F83E0((__int64)v75, v97, v96, v74, 0);
  v77 = a1[1];
  if ( v77 )
  {
    v78 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v77 + 40, (__int64)v76);
    v79 = v76[3];
    v80 = *v78;
    v76[4] = v78;
    v80 &= 0xFFFFFFFFFFFFFFF8LL;
    v76[3] = v80 | v79 & 7;
    *(_QWORD *)(v80 + 8) = v76 + 3;
    *v78 = *v78 & 7 | (unsigned __int64)(v76 + 3);
  }
  sub_164B780((__int64)v76, v106);
  v81 = *a1;
  if ( *a1 )
  {
    v103[0] = *a1;
    sub_1623A60((__int64)v103, v81, 2);
    v82 = v76[6];
    if ( v82 )
      sub_161E7C0((__int64)(v76 + 6), v82);
    v83 = (unsigned __int8 *)v103[0];
    v76[6] = v103[0];
    if ( v83 )
      sub_1623210((__int64)v103, v83, (__int64)(v76 + 6));
  }
  v84 = *(_QWORD *)(v97 + 48);
  a1[1] = v97;
  a1[2] = v84;
  if ( v84 != v97 + 40 )
  {
    if ( !v84 )
      BUG();
    v85 = *(_QWORD *)(v84 + 24);
    v106[0] = v85;
    if ( v85 )
    {
      sub_1623A60((__int64)v106, v85, 2);
      v86 = *a1;
      if ( !*a1 )
        goto LABEL_74;
    }
    else
    {
      v86 = *a1;
      if ( !*a1 )
        return v101;
    }
    sub_161E7C0((__int64)a1, v86);
LABEL_74:
    v87 = (unsigned __int8 *)v106[0];
    *a1 = v106[0];
    if ( v87 )
      sub_1623210((__int64)v106, v87, (__int64)a1);
  }
  return v101;
}
