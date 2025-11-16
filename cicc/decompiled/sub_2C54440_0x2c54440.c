// Function: sub_2C54440
// Address: 0x2c54440
//
__int64 __fastcall sub_2C54440(__int64 a1, unsigned __int8 *a2)
{
  int v4; // eax
  __int64 result; // rax
  unsigned __int8 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // rbx
  __int64 v11; // r14
  _DWORD *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r8
  unsigned int v16; // ecx
  int v17; // edx
  unsigned int v18; // eax
  unsigned __int64 v19; // r8
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rdi
  signed __int64 v24; // rax
  unsigned int v25; // r11d
  int v26; // edx
  _BOOL4 v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // r10
  bool v30; // r15
  __int64 v31; // rax
  int v32; // edx
  unsigned __int64 v33; // rax
  __int64 v34; // rbx
  int v35; // edx
  __int64 v36; // rax
  int v37; // edx
  unsigned __int64 v38; // rax
  bool v39; // sf
  bool v40; // of
  unsigned __int64 v41; // rax
  _BYTE *v42; // r14
  unsigned __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // r11
  _DWORD *v46; // r10
  __int64 (__fastcall *v47)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v48; // rax
  __int64 v49; // r15
  __int64 v50; // r13
  __int64 i; // rbx
  char v52; // al
  unsigned __int64 v53; // rax
  __int64 v54; // rax
  _BYTE *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  int v59; // edx
  _QWORD *v60; // rax
  __int64 v61; // r14
  __int64 v62; // rbx
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // rax
  __int64 v66; // [rsp+20h] [rbp-130h]
  unsigned __int64 v67; // [rsp+20h] [rbp-130h]
  unsigned __int64 v68; // [rsp+20h] [rbp-130h]
  __int64 v69; // [rsp+20h] [rbp-130h]
  _BYTE *v70; // [rsp+28h] [rbp-128h]
  __int64 *v71; // [rsp+38h] [rbp-118h]
  unsigned int v72; // [rsp+38h] [rbp-118h]
  __int64 *v73; // [rsp+40h] [rbp-110h]
  __int64 v74; // [rsp+40h] [rbp-110h]
  int v75; // [rsp+40h] [rbp-110h]
  unsigned int v76; // [rsp+48h] [rbp-108h]
  unsigned __int64 v77; // [rsp+48h] [rbp-108h]
  unsigned __int64 v78; // [rsp+48h] [rbp-108h]
  __int64 **v79; // [rsp+48h] [rbp-108h]
  unsigned __int64 v80; // [rsp+50h] [rbp-100h]
  unsigned int v81; // [rsp+58h] [rbp-F8h]
  char v82; // [rsp+58h] [rbp-F8h]
  int v83; // [rsp+58h] [rbp-F8h]
  _DWORD *v84; // [rsp+58h] [rbp-F8h]
  __int64 v85; // [rsp+58h] [rbp-F8h]
  _DWORD *v86; // [rsp+58h] [rbp-F8h]
  __int64 v87; // [rsp+60h] [rbp-F0h]
  __int64 v88; // [rsp+60h] [rbp-F0h]
  void *v89; // [rsp+60h] [rbp-F0h]
  __int64 v90; // [rsp+60h] [rbp-F0h]
  char *v91; // [rsp+68h] [rbp-E8h]
  _BYTE *v92; // [rsp+68h] [rbp-E8h]
  unsigned __int8 v93; // [rsp+68h] [rbp-E8h]
  int v94[8]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v95; // [rsp+90h] [rbp-C0h]
  _QWORD v96[4]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v97; // [rsp+C0h] [rbp-90h]
  _DWORD *v98; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v99; // [rsp+D8h] [rbp-78h]
  _BYTE v100[112]; // [rsp+E0h] [rbp-70h] BYREF

  v4 = *a2;
  if ( (unsigned __int8)v4 <= 0x1Cu )
  {
    if ( *((_WORD *)a2 + 1) != 49 )
      return 0;
  }
  else if ( v4 != 78 )
  {
    return 0;
  }
  if ( (a2[7] & 0x40) != 0 )
    v6 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v6 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)(*(_QWORD *)v6 + 16LL);
  if ( !v8 )
    return 0;
  if ( *(_QWORD *)(v8 + 8) )
    return 0;
  if ( *(_BYTE *)v7 != 92 )
    return 0;
  v80 = *(_QWORD *)(v7 - 64);
  if ( !v80 )
    return 0;
  v70 = *(_BYTE **)(v7 - 32);
  if ( !v70 )
    return 0;
  v9 = *(_DWORD *)(v7 + 80);
  v10 = *((_QWORD *)a2 + 1);
  v91 = *(char **)(v7 + 72);
  v87 = v9;
  if ( *(_BYTE *)(v10 + 8) != 17 )
    return 0;
  v11 = *(_QWORD *)(v80 + 8);
  if ( *(_BYTE *)(v11 + 8) != 17 )
    return 0;
  v73 = (__int64 *)*((_QWORD *)a2 + 1);
  v71 = *(__int64 **)(v80 + 8);
  v81 = sub_BCB060(v10);
  v76 = sub_BCB060(v11);
  v12 = (_DWORD *)sub_BCAE30(v11);
  v99 = v13;
  v98 = v12;
  v14 = sub_CA1930(&v98);
  v15 = v81;
  v16 = v81;
  if ( v14 % v81 )
    return 0;
  v82 = 1;
  v17 = (unsigned __int8)*v70;
  if ( (unsigned int)(v17 - 12) > 1 )
  {
    v53 = v80;
    while ( *(_BYTE *)v53 == 78 )
    {
      v53 = *(_QWORD *)(v53 - 32);
      if ( !v53 )
        BUG();
    }
    v54 = *(_QWORD *)(v53 + 8);
    if ( *(_BYTE *)(v54 + 8) != 17 )
      v54 = 0;
    v69 = v54;
    v55 = v70;
    while ( (_BYTE)v17 == 78 )
    {
      v55 = (_BYTE *)*((_QWORD *)v55 - 4);
      if ( !v55 )
        BUG();
      LOBYTE(v17) = *v55;
    }
    v56 = *((_QWORD *)v55 + 1);
    if ( *(_BYTE *)(v56 + 8) == 17 )
    {
      v57 = *(_QWORD *)(v10 + 24);
      if ( (!v69 || v57 != *(_QWORD *)(v69 + 24)) && v57 != *(_QWORD *)(v56 + 24) )
        return 0;
    }
    else if ( !v69 || *(_QWORD *)(v69 + 24) != *(_QWORD *)(v10 + 24) )
    {
      return 0;
    }
    v82 = 0;
  }
  v98 = v100;
  v99 = 0x1000000000LL;
  v18 = v76;
  if ( v16 > v76 )
  {
    v68 = v15;
    v52 = sub_9B8470(v16 / v76, v91, v9, (__int64)&v98);
    v19 = v68;
    if ( !v52 )
      goto LABEL_58;
  }
  else
  {
    v77 = v15;
    sub_9B8300(v18 / v16, (unsigned int *)v91, v9, (__int64)&v98);
    v19 = v77;
  }
  v78 = v19;
  v96[0] = sub_BCAE30(v11);
  v96[1] = v20;
  v21 = sub_CA1930(v96) / v78;
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v73 = **(__int64 ***)(v10 + 16);
  v79 = (__int64 **)sub_BCDA70(v73, v21);
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    v71 = **(__int64 ***)(v11 + 16);
  v22 = sub_BCDA70(v71, v9);
  v23 = *(__int64 **)(a1 + 152);
  v74 = v22;
  if ( !v82 )
  {
    v24 = sub_DFD060(v23, 49, (__int64)v79, v11);
    v25 = 6;
    v27 = v26 == 1;
    v28 = 2 * v24;
    v83 = v27;
    if ( !is_mul_ok(2u, v24) )
    {
      v28 = 0x8000000000000000LL;
      if ( v24 > 0 )
        v28 = 0x7FFFFFFFFFFFFFFFLL;
    }
    goto LABEL_27;
  }
  v58 = sub_DFD060(v23, 49, (__int64)v79, v11);
  v83 = v59;
  if ( v59 != 1 )
  {
    v83 = 0;
    v28 = v58;
    v25 = 7;
LABEL_27:
    v29 = v28;
    goto LABEL_28;
  }
  v29 = v58;
  v25 = 7;
LABEL_28:
  v30 = 1;
  v66 = v29;
  v72 = v25;
  v31 = sub_DFBC30(
          *(__int64 **)(a1 + 152),
          v25,
          (__int64)v79,
          (__int64)v98,
          (unsigned int)v99,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          0,
          0,
          0);
  if ( v83 != 1 )
  {
    v83 = v32;
    v30 = v32 != 0;
  }
  v40 = __OFADD__(v66, v31);
  v33 = v66 + v31;
  if ( v40 )
  {
    v33 = 0x8000000000000000LL;
    if ( v66 > 0 )
      v33 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v67 = v33;
  v34 = sub_DFD060(*(__int64 **)(a1 + 152), 49, v10, v74);
  v75 = v35;
  v36 = sub_DFBC30(*(__int64 **)(a1 + 152), v72, v11, (__int64)v91, v87, *(unsigned int *)(a1 + 192), 0, 0, 0, 0, 0);
  if ( v75 == 1 )
    v37 = 1;
  v40 = __OFADD__(v34, v36);
  v38 = v34 + v36;
  if ( v40 )
  {
    v38 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v34 <= 0 )
      v38 = 0x8000000000000000LL;
  }
  v40 = __OFSUB__(v37, v83);
  v39 = v37 - v83 < 0;
  if ( v37 == v83 )
  {
    v40 = __OFSUB__(v38, v67);
    v39 = (__int64)(v38 - v67) < 0;
  }
  if ( v39 == v40 && !v30 )
  {
    v97 = 257;
    while ( *(_BYTE *)v80 == 78 )
    {
      v80 = *(_QWORD *)(v80 - 32);
      if ( !v80 )
        BUG();
    }
    v41 = sub_2C511B0((__int64 *)(a1 + 8), 0x31u, v80, v79, (__int64)v96, 0, v94[0], 0);
    v97 = 257;
    v42 = (_BYTE *)v41;
    while ( *v70 == 78 )
    {
      v70 = (_BYTE *)*((_QWORD *)v70 - 4);
      if ( !v70 )
        BUG();
    }
    v43 = sub_2C511B0((__int64 *)(a1 + 8), 0x31u, (unsigned __int64)v70, v79, (__int64)v96, 0, v94[0], 0);
    v44 = *(_QWORD *)(a1 + 88);
    v45 = (unsigned int)v99;
    v95 = 257;
    v92 = (_BYTE *)v43;
    v46 = v98;
    v47 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v44 + 112LL);
    if ( v47 == sub_9B6630 )
    {
      if ( *v42 > 0x15u || *v92 > 0x15u )
        goto LABEL_80;
      v84 = v98;
      v88 = (unsigned int)v99;
      v48 = sub_AD5CE0((__int64)v42, (__int64)v92, v98, (unsigned int)v99, 0);
      v45 = v88;
      v46 = v84;
      v49 = v48;
    }
    else
    {
      v86 = v98;
      v90 = (unsigned int)v99;
      v65 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v47)(
              v44,
              v42,
              v92,
              v98,
              (unsigned int)v99);
      v46 = v86;
      v45 = v90;
      v49 = v65;
    }
    if ( v49 )
    {
LABEL_47:
      v50 = a1 + 200;
      sub_BD84D0((__int64)a2, v49);
      if ( *(_BYTE *)v49 > 0x1Cu )
      {
        sub_BD6B90((unsigned __int8 *)v49, a2);
        for ( i = *(_QWORD *)(v49 + 16); i; i = *(_QWORD *)(i + 8) )
          sub_F15FC0(v50, *(_QWORD *)(i + 24));
        if ( *(_BYTE *)v49 > 0x1Cu )
          sub_F15FC0(v50, v49);
      }
      result = 1;
      if ( *a2 > 0x1Cu )
      {
        sub_F15FC0(v50, (__int64)a2);
        result = 1;
      }
      goto LABEL_54;
    }
LABEL_80:
    v85 = v45;
    v89 = v46;
    v97 = 257;
    v60 = sub_BD2C40(112, unk_3F1FE60);
    v49 = (__int64)v60;
    if ( v60 )
      sub_B4E9E0((__int64)v60, (__int64)v42, (__int64)v92, v89, v85, (__int64)v96, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
      *(_QWORD *)(a1 + 96),
      v49,
      v94,
      *(_QWORD *)(a1 + 64),
      *(_QWORD *)(a1 + 72));
    v61 = *(_QWORD *)(a1 + 8);
    v62 = v61 + 16LL * *(unsigned int *)(a1 + 16);
    while ( v62 != v61 )
    {
      v63 = *(_QWORD *)(v61 + 8);
      v64 = *(_DWORD *)v61;
      v61 += 16;
      sub_B99FD0(v49, v64, v63);
    }
    goto LABEL_47;
  }
LABEL_58:
  result = 0;
LABEL_54:
  if ( v98 != (_DWORD *)v100 )
  {
    v93 = result;
    _libc_free((unsigned __int64)v98);
    return v93;
  }
  return result;
}
