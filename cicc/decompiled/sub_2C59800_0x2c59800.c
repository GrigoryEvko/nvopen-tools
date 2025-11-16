// Function: sub_2C59800
// Address: 0x2c59800
//
__int64 __fastcall sub_2C59800(__int64 a1, unsigned __int8 *a2)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r10
  _QWORD *v17; // rdi
  int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r10
  int v22; // edx
  __int64 *v23; // rax
  __int64 *v24; // r10
  int v25; // r11d
  __int64 v26; // rax
  int v27; // r11d
  int v28; // edx
  __int64 v29; // r10
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rcx
  int v33; // esi
  int v34; // r11d
  __int64 v35; // rcx
  unsigned __int64 v36; // r10
  _BOOL4 v37; // r8d
  __int64 v38; // rcx
  __int64 v39; // r14
  int v40; // edi
  __int64 v41; // r9
  __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rdx
  bool v45; // al
  unsigned __int64 v46; // rdx
  bool v47; // sf
  bool v48; // of
  unsigned __int8 *v49; // r14
  const char *v50; // rdx
  _QWORD *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int8 *v55; // r9
  __int64 (__fastcall *v56)(__int64, _BYTE *, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v57; // rax
  __int64 v58; // r12
  __int64 v59; // r13
  __int64 j; // rbx
  __int64 i; // rax
  __int64 v62; // rdx
  unsigned __int8 **v63; // rdx
  __int64 v64; // rax
  int v65; // edx
  __int64 v66; // rax
  int v67; // edx
  char v68; // al
  _QWORD *v69; // rax
  __int64 v70; // r9
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 v76; // rax
  __int64 v77; // rax
  unsigned __int8 *v78; // rax
  __int64 v79; // rax
  unsigned __int8 *v80; // rax
  unsigned __int64 v81; // rdi
  int v82; // [rsp+4h] [rbp-10Ch]
  __int64 *v83; // [rsp+8h] [rbp-108h]
  int v84; // [rsp+8h] [rbp-108h]
  __int64 *v85; // [rsp+10h] [rbp-100h]
  int v86; // [rsp+10h] [rbp-100h]
  __int64 v87; // [rsp+10h] [rbp-100h]
  int v88; // [rsp+10h] [rbp-100h]
  __int64 v89; // [rsp+18h] [rbp-F8h]
  __int64 v90; // [rsp+18h] [rbp-F8h]
  int v91; // [rsp+20h] [rbp-F0h]
  unsigned int v92; // [rsp+24h] [rbp-ECh]
  unsigned __int8 *v93; // [rsp+28h] [rbp-E8h]
  __int64 v94; // [rsp+30h] [rbp-E0h]
  __int64 v95; // [rsp+38h] [rbp-D8h]
  char v96; // [rsp+40h] [rbp-D0h]
  __int64 v97; // [rsp+40h] [rbp-D0h]
  __int64 v98; // [rsp+48h] [rbp-C8h]
  bool v99; // [rsp+48h] [rbp-C8h]
  int v100; // [rsp+48h] [rbp-C8h]
  __int64 *v101; // [rsp+48h] [rbp-C8h]
  unsigned __int8 *v102; // [rsp+50h] [rbp-C0h]
  __int64 v103; // [rsp+50h] [rbp-C0h]
  __int64 v104; // [rsp+58h] [rbp-B8h]
  _BYTE *v105; // [rsp+58h] [rbp-B8h]
  int v106; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v107; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v108; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v109; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v110; // [rsp+60h] [rbp-B0h]
  __int64 v111; // [rsp+70h] [rbp-A0h]
  __int64 v112; // [rsp+78h] [rbp-98h]
  int v113[8]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v114; // [rsp+A0h] [rbp-70h]
  const char *v115[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v116; // [rsp+D0h] [rbp-40h]

  v4 = *a2;
  if ( (unsigned int)(v4 - 42) > 0x11 )
  {
    if ( (unsigned __int8)(v4 - 82) > 1u )
      return 0;
    v5 = *((_QWORD *)a2 - 8);
    if ( !v5 )
      return 0;
    v6 = *((_QWORD *)a2 - 4);
    if ( !v6 )
      return 0;
    v7 = sub_B53900((__int64)a2);
    if ( v7 != 42 )
    {
      for ( i = *((_QWORD *)a2 + 2); i; i = *(_QWORD *)(i + 8) )
      {
        v62 = *(_QWORD *)(i + 24);
        if ( *(_BYTE *)v62 == 86 )
        {
          v63 = (*(_BYTE *)(v62 + 7) & 0x40) != 0
              ? *(unsigned __int8 ***)(v62 - 8)
              : (unsigned __int8 **)(v62 - 32LL * (*(_DWORD *)(v62 + 4) & 0x7FFFFFF));
          if ( *v63 == a2 )
            return 0;
        }
      }
    }
  }
  else
  {
    v5 = *((_QWORD *)a2 - 8);
    if ( !v5 )
      return 0;
    v6 = *((_QWORD *)a2 - 4);
    if ( !v6 )
      return 0;
    v7 = 42;
  }
  if ( *(_BYTE *)v5 == 91 )
  {
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
    {
      v9 = *(_QWORD *)(v5 - 8);
      v102 = *(unsigned __int8 **)v9;
      if ( **(_BYTE **)v9 > 0x15u )
        return 0;
    }
    else
    {
      v9 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
      v102 = *(unsigned __int8 **)v9;
      if ( **(_BYTE **)v9 > 0x15u )
        return 0;
    }
    v104 = *(_QWORD *)(v9 + 32);
    if ( !v104 )
      return 0;
    v10 = *(_QWORD *)(v9 + 64);
    if ( *(_BYTE *)v10 != 17 )
      return 0;
    if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    {
      v107 = *(_QWORD *)(v10 + 24);
    }
    else
    {
      v98 = *(_QWORD *)(v9 + 64);
      v106 = *(_DWORD *)(v10 + 32);
      if ( v106 - (unsigned int)sub_C444A0(v10 + 24) > 0x40 )
        return 0;
      v107 = **(_QWORD **)(v98 + 24);
    }
    if ( *(_BYTE *)v6 != 91 )
    {
      if ( *(_BYTE *)v6 <= 0x15u )
      {
        v93 = (unsigned __int8 *)v6;
        v11 = 0;
        v12 = 0;
        v94 = 0;
        v96 = 1;
        v99 = 0;
        goto LABEL_30;
      }
      return 0;
    }
  }
  else
  {
    if ( *(_BYTE *)v5 > 0x15u || *(_BYTE *)v6 != 91 )
      return 0;
    v102 = (unsigned __int8 *)v5;
    v104 = 0;
    v107 = 0;
  }
  if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
  {
    v13 = *(_QWORD *)(v6 - 8);
    v93 = *(unsigned __int8 **)v13;
    if ( **(_BYTE **)v13 > 0x15u )
      return 0;
    v94 = *(_QWORD *)(v13 + 32);
    if ( !v94 )
      return 0;
  }
  else
  {
    v13 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
    v93 = *(unsigned __int8 **)v13;
    if ( **(_BYTE **)v13 > 0x15u )
      return 0;
    v94 = *(_QWORD *)(v13 + 32);
    if ( !v94 )
      return 0;
  }
  v14 = *(_QWORD *)(v13 + 64);
  if ( *(_BYTE *)v14 != 17 )
    return 0;
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
  {
    v11 = *(_QWORD *)(v14 + 24);
    goto LABEL_27;
  }
  v97 = *(_QWORD *)(v13 + 64);
  v100 = *(_DWORD *)(v14 + 32);
  if ( v100 - (unsigned int)sub_C444A0(v14 + 24) > 0x40 )
    return 0;
  v11 = **(_QWORD **)(v97 + 24);
LABEL_27:
  v99 = v104 == 0;
  if ( v104 && v107 != v11 )
    return 0;
  v96 = 0;
  v12 = 1;
LABEL_30:
  if ( *(unsigned int *)(*(_QWORD *)(v5 + 8) + 32LL) <= v107 || *(unsigned int *)(*(_QWORD *)(v6 + 8) + 32LL) <= v11 )
    return 0;
  if ( v99 )
  {
    if ( *(_BYTE *)v94 > 0x1Cu )
    {
      if ( v12 )
      {
        v109 = v11;
        v68 = sub_B46420(v94);
        v11 = v109;
        if ( v68 )
          return 0;
      }
    }
    v107 = v11;
    v15 = *(_QWORD *)(v94 + 8);
  }
  else
  {
    if ( *(_BYTE *)v104 > 0x1Cu && v96 && (unsigned __int8)sub_B46420(v104) )
      return 0;
    v15 = *(_QWORD *)(v104 + 8);
  }
  v16 = *(_QWORD *)(a1 + 152);
  v95 = *((_QWORD *)a2 + 1);
  v92 = *a2 - 29;
  if ( v7 == 42 )
  {
    v64 = sub_DFD800(v16, v92, v15, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
    v88 = v65;
    v90 = v64;
    v66 = sub_DFD800(*(_QWORD *)(a1 + 152), v92, v95, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
    v27 = v88;
    v91 = v67;
    v29 = v66;
  }
  else
  {
    v17 = *(_QWORD **)v15;
    v83 = (__int64 *)v16;
    v18 = *(unsigned __int8 *)(v15 + 8);
    v89 = v15;
    if ( (unsigned int)(v18 - 17) > 1 )
    {
      sub_BCB2A0(v17);
      v21 = v83;
      v20 = v89;
    }
    else
    {
      BYTE4(v112) = (_BYTE)v18 == 18;
      LODWORD(v112) = *(_DWORD *)(v15 + 32);
      v19 = (__int64 *)sub_BCB2A0(v17);
      sub_BCE1B0(v19, v112);
      v20 = v89;
      v21 = v83;
    }
    v90 = sub_DFD2D0(v21, v92, v20);
    v82 = v22;
    v85 = *(__int64 **)(a1 + 152);
    if ( (unsigned int)*(unsigned __int8 *)(v95 + 8) - 17 > 1 )
    {
      sub_BCB2A0(*(_QWORD **)v95);
      v25 = v82;
      v24 = v85;
    }
    else
    {
      BYTE4(v111) = *(_BYTE *)(v95 + 8) == 18;
      LODWORD(v111) = *(_DWORD *)(v95 + 32);
      v23 = (__int64 *)sub_BCB2A0(*(_QWORD **)v95);
      sub_BCE1B0(v23, v111);
      v24 = v85;
      v25 = v82;
    }
    v86 = v25;
    v26 = sub_DFD2D0(v24, v92, v95);
    v27 = v86;
    v91 = v28;
    v29 = v26;
  }
  v84 = v27;
  v87 = v29;
  v30 = sub_DFD330(*(__int64 **)(a1 + 152));
  v32 = v30;
  v33 = v31;
  v34 = v84;
  if ( !v96 )
  {
    if ( v99 )
    {
      if ( v31 == 1 )
      {
        v33 = 1;
        v32 = v30;
      }
      else
      {
        v33 = 0;
      }
    }
    else
    {
      v32 = 2 * v30;
      if ( __OFADD__(v30, v30) )
      {
        v32 = 0x8000000000000000LL;
        if ( v30 > 0 )
          v32 = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  if ( v91 == 1 )
    v33 = 1;
  v48 = __OFADD__(v87, v32);
  v35 = v87 + v32;
  if ( v48 )
  {
    v81 = 0x8000000000000000LL;
    if ( v87 > 0 )
      v81 = 0x7FFFFFFFFFFFFFFFLL;
    v36 = v81;
  }
  else
  {
    v36 = v35;
  }
  v37 = v31 == 1;
  if ( v96 )
  {
    v39 = 0;
    v40 = 0;
  }
  else
  {
    v38 = *(_QWORD *)(v6 + 16);
    v39 = v30;
    if ( v38 )
      v39 = v30 * (*(_QWORD *)(v38 + 8) != 0);
    v40 = v31 == 1;
    if ( v99 )
    {
      v37 = 0;
      v42 = 0;
      goto LABEL_57;
    }
  }
  v41 = *(_QWORD *)(v5 + 16);
  v42 = v30;
  if ( v41 )
    v42 = v30 * (*(_QWORD *)(v41 + 8) != 0);
LABEL_57:
  if ( v31 == 1 )
    v34 = 1;
  v43 = v30 + v90;
  if ( __OFADD__(v30, v90) )
  {
    v43 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v30 <= 0 )
      v43 = 0x8000000000000000LL;
  }
  if ( v37 )
    v34 = 1;
  v48 = __OFADD__(v42, v43);
  v44 = v42 + v43;
  if ( v48 )
  {
    v44 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v42 <= 0 )
      v44 = 0x8000000000000000LL;
  }
  v45 = 1;
  if ( v40 != 1 )
  {
    v40 = v34;
    v45 = v34 != 0;
  }
  v48 = __OFADD__(v39, v44);
  v46 = v39 + v44;
  if ( v48 )
  {
    v46 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v39 <= 0 )
      v46 = 0x8000000000000000LL;
  }
  v48 = __OFSUB__(v33, v40);
  v47 = v33 - v40 < 0;
  if ( v33 == v40 )
  {
    v48 = __OFSUB__(v36, v46);
    v47 = (__int64)(v36 - v46) < 0;
  }
  if ( v47 != v48 || v45 )
    return 0;
  if ( v99 )
  {
    v79 = sub_BCB2E0(*(_QWORD **)(a1 + 80));
    v80 = (unsigned __int8 *)sub_ACD640(v79, v107, 0);
    v104 = sub_AD5840((__int64)v102, v80, 0);
  }
  else if ( v96 )
  {
    v77 = sub_BCB2E0(*(_QWORD **)(a1 + 80));
    v78 = (unsigned __int8 *)sub_ACD640(v77, v107, 0);
    v94 = sub_AD5840((__int64)v93, v78, 0);
  }
  v101 = (__int64 *)(a1 + 8);
  v116 = 257;
  if ( v7 == 42 )
    v49 = (unsigned __int8 *)sub_2C51350(
                               v101,
                               v92,
                               (unsigned __int8 *)v104,
                               (unsigned __int8 *)v94,
                               v113[0],
                               0,
                               (__int64)v115,
                               0);
  else
    v49 = (unsigned __int8 *)sub_2B22A00(a1 + 8, v7, v104, v94, (__int64)v115, 0);
  v115[0] = sub_BD5D20((__int64)a2);
  v116 = 773;
  v115[1] = v50;
  v115[2] = ".scalar";
  sub_BD6B50(v49, v115);
  if ( *v49 > 0x1Cu )
    sub_B45260(v49, (__int64)a2, 1);
  v116 = 257;
  if ( v7 == 42 )
    v105 = (_BYTE *)sub_2C51350(v101, v92, v102, v93, v113[0], 0, (__int64)v115, 0);
  else
    v105 = (_BYTE *)sub_2B22A00((__int64)v101, v7, (__int64)v102, (__int64)v93, (__int64)v115, 0);
  v51 = *(_QWORD **)(a1 + 80);
  v114 = 257;
  v52 = sub_BCB2E0(v51);
  v53 = sub_ACD640(v52, v107, 0);
  v54 = *(_QWORD *)(a1 + 88);
  v55 = (unsigned __int8 *)v53;
  v56 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v54 + 104LL);
  if ( v56 == sub_948040 )
  {
    if ( *v105 > 0x15u || *v49 > 0x15u || *v55 > 0x15u )
      goto LABEL_126;
    v108 = v55;
    v57 = sub_AD5A90((__int64)v105, v49, v55, 0);
    v55 = v108;
    v58 = v57;
  }
  else
  {
    v110 = v55;
    v76 = v56(v54, v105, v49, v55);
    v55 = v110;
    v58 = v76;
  }
  if ( !v58 )
  {
LABEL_126:
    v116 = 257;
    v103 = (__int64)v55;
    v69 = sub_BD2C40(72, 3u);
    v70 = v103;
    v58 = (__int64)v69;
    if ( v69 )
      sub_B4DFA0((__int64)v69, (__int64)v105, (__int64)v49, v103, (__int64)v115, v103, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 96) + 16LL))(
      *(_QWORD *)(a1 + 96),
      v58,
      v113,
      *(_QWORD *)(a1 + 64),
      *(_QWORD *)(a1 + 72),
      v70);
    v71 = *(_QWORD *)(a1 + 8);
    v72 = 16LL * *(unsigned int *)(a1 + 16);
    v73 = v71 + v72;
    while ( v73 != v71 )
    {
      v74 = *(_QWORD *)(v71 + 8);
      v75 = *(_DWORD *)v71;
      v71 += 16;
      sub_B99FD0(v58, v75, v74);
    }
  }
  v59 = a1 + 200;
  sub_BD84D0((__int64)a2, v58);
  if ( *(_BYTE *)v58 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)v58, a2);
    for ( j = *(_QWORD *)(v58 + 16); j; j = *(_QWORD *)(j + 8) )
      sub_F15FC0(v59, *(_QWORD *)(j + 24));
    if ( *(_BYTE *)v58 > 0x1Cu )
      sub_F15FC0(v59, v58);
  }
  result = 1;
  if ( *a2 > 0x1Cu )
  {
    sub_F15FC0(v59, (__int64)a2);
    return 1;
  }
  return result;
}
