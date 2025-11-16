// Function: sub_27C62F0
// Address: 0x27c62f0
//
__int64 __fastcall sub_27C62F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // r12
  __int64 *v16; // r12
  __int64 *v17; // r8
  _QWORD *v18; // r13
  unsigned __int64 v19; // r12
  int v20; // eax
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned __int64 v24; // rbx
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r12
  __int64 v33; // rax
  int v34; // ecx
  _QWORD *v35; // rdx
  unsigned int v36; // r12d
  __int64 v37; // r9
  _QWORD *v38; // r14
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // eax
  _QWORD *v44; // rdi
  __int64 *v46; // r12
  _QWORD *v47; // rsi
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rsi
  __int64 v54; // rax
  _QWORD **v55; // rdx
  int v56; // ecx
  __int64 *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r13
  _BYTE *v60; // r12
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  unsigned __int64 *v68; // r13
  unsigned __int64 *v69; // rdi
  unsigned __int64 v70; // rdi
  int v71; // r12d
  __int64 v72; // r12
  __int64 v73; // r14
  __int64 v74; // rbx
  _BYTE *v75; // r14
  __int64 v76; // r12
  __int64 v77; // rdx
  unsigned int v78; // esi
  unsigned __int64 v79; // rsi
  _QWORD *v80; // rax
  _BYTE *v81; // r12
  __int64 v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 v85; // [rsp+8h] [rbp-168h]
  unsigned __int64 v86; // [rsp+8h] [rbp-168h]
  __int64 *v87; // [rsp+10h] [rbp-160h]
  unsigned __int64 v88; // [rsp+10h] [rbp-160h]
  char v90; // [rsp+20h] [rbp-150h]
  __int64 *v91; // [rsp+20h] [rbp-150h]
  __int64 v92; // [rsp+28h] [rbp-148h]
  unsigned int v93; // [rsp+28h] [rbp-148h]
  __int64 v95; // [rsp+38h] [rbp-138h]
  __int64 v96; // [rsp+38h] [rbp-138h]
  unsigned __int64 v97; // [rsp+38h] [rbp-138h]
  __int64 v98; // [rsp+48h] [rbp-128h]
  _QWORD v99[4]; // [rsp+50h] [rbp-120h] BYREF
  char v100; // [rsp+70h] [rbp-100h]
  char v101; // [rsp+71h] [rbp-FFh]
  unsigned __int64 v102[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v103; // [rsp+A0h] [rbp-D0h]
  _BYTE *v104; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v105; // [rsp+B8h] [rbp-B8h]
  _BYTE v106[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v107; // [rsp+E0h] [rbp-90h]
  __int64 v108; // [rsp+E8h] [rbp-88h]
  __int64 v109; // [rsp+F0h] [rbp-80h]
  __int64 v110; // [rsp+F8h] [rbp-78h]
  void **v111; // [rsp+100h] [rbp-70h]
  void **v112; // [rsp+108h] [rbp-68h]
  __int64 v113; // [rsp+110h] [rbp-60h]
  int v114; // [rsp+118h] [rbp-58h]
  __int16 v115; // [rsp+11Ch] [rbp-54h]
  char v116; // [rsp+11Eh] [rbp-52h]
  __int64 v117; // [rsp+120h] [rbp-50h]
  __int64 v118; // [rsp+128h] [rbp-48h]
  void *v119; // [rsp+130h] [rbp-40h] BYREF
  void *v120; // [rsp+138h] [rbp-38h] BYREF

  v10 = sub_D47930(a2);
  if ( (*(_DWORD *)(a5 + 4) & 0x7FFFFFF) != 0 )
  {
    v11 = *(_QWORD *)(a5 - 8);
    v12 = v10;
    v13 = 0;
    do
    {
      if ( v12 == *(_QWORD *)(v11 + 32LL * *(unsigned int *)(a5 + 72) + 8 * v13) )
      {
        v14 = 32 * v13;
        goto LABEL_6;
      }
      ++v13;
    }
    while ( (*(_DWORD *)(a5 + 4) & 0x7FFFFFF) != (_DWORD)v13 );
    v14 = 0x1FFFFFFFE0LL;
  }
  else
  {
    v14 = 0x1FFFFFFFE0LL;
    v11 = *(_QWORD *)(a5 - 8);
  }
LABEL_6:
  v15 = *(unsigned __int8 **)(v11 + v14);
  v92 = a3 + 48;
  if ( a3 != sub_D47930(a2) )
    goto LABEL_7;
  if ( *(_BYTE *)(*(_QWORD *)(a5 + 8) + 8LL) == 12 )
  {
    v95 = (__int64)v15;
    v90 = 1;
  }
  else
  {
    v95 = (__int64)v15;
    v90 = sub_27C10F0((__int64)v15, a3);
    if ( !v90 )
    {
      v51 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v51 == v50 )
      {
        v53 = 0;
      }
      else
      {
        if ( !v51 )
          BUG();
        v52 = *(unsigned __int8 *)(v51 - 24);
        v53 = 0;
        v54 = v51 - 24;
        if ( (unsigned int)(v52 - 30) < 0xB )
          v53 = v54;
      }
      v95 = (__int64)v15;
      v90 = sub_98EF90((__int64)v15, v53, *(_QWORD *)(a1 + 16), v50, *(_QWORD *)(a1 + 16));
      if ( !v90 )
      {
LABEL_7:
        v95 = a5;
        v90 = 0;
      }
    }
  }
  if ( (unsigned int)*v15 - 42 <= 0x11 )
  {
    v87 = sub_DD8400(*(_QWORD *)(a1 + 8), (__int64)v15);
    if ( sub_B448F0((__int64)v15) )
      sub_B447F0(v15, (*((_WORD *)v87 + 14) & 2) != 0);
    if ( sub_B44900((__int64)v15) )
      sub_B44850(v15, (*((_WORD *)v87 + 14) & 4) != 0);
  }
  v16 = *(__int64 **)(a1 + 8);
  v17 = sub_DD8400((__int64)v16, a5);
  if ( *(_BYTE *)(*(_QWORD *)(a5 + 8) + 8LL) == 12 )
  {
    v85 = (__int64)v17;
    v63 = sub_D95540(*(_QWORD *)v17[4]);
    v88 = sub_D97050((__int64)v16, v63);
    v64 = sub_D95540(a4);
    v65 = sub_D97050((__int64)v16, v64);
    v17 = (__int64 *)v85;
    if ( v88 > v65 && (*(_WORD *)(**(_QWORD **)(v85 + 32) + 24LL) || *(_WORD *)(a4 + 24)) )
    {
      v66 = sub_D95540(a4);
      v17 = sub_DC5200((__int64)v16, v85, v66, 0);
    }
  }
  if ( v90 )
    v17 = sub_DCC620((__int64)v17, v16);
  v91 = v17;
  v18 = sub_DD0540((__int64)v17, a4, v16);
  v19 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 == v92 )
  {
    v21 = 0;
  }
  else
  {
    if ( !v19 )
      BUG();
    v20 = *(unsigned __int8 *)(v19 - 24);
    v21 = v19 - 24;
    if ( (unsigned int)(v20 - 30) >= 0xB )
      v21 = 0;
  }
  v22 = sub_D95540(*(_QWORD *)v91[4]);
  v23 = (__int64)sub_F8DB90(a6, (__int64)v18, v22, v21 + 24, 0);
  v24 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 == v92 )
    goto LABEL_128;
  if ( !v24 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA )
LABEL_128:
    BUG();
  v25 = *(_QWORD *)(v24 - 56);
  if ( *(_BYTE *)(a2 + 84) )
  {
    v26 = *(_QWORD **)(a2 + 64);
    v27 = &v26[*(unsigned int *)(a2 + 76)];
    if ( v26 == v27 )
      goto LABEL_58;
    while ( v25 != *v26 )
    {
      if ( v27 == ++v26 )
        goto LABEL_58;
    }
LABEL_28:
    v93 = 33;
    goto LABEL_29;
  }
  if ( sub_C8CA60(a2 + 56, v25) )
    goto LABEL_28;
LABEL_58:
  v93 = 32;
LABEL_29:
  v116 = 7;
  v110 = sub_BD5C60(v24 - 24);
  v111 = &v119;
  v112 = &v120;
  v104 = v106;
  v119 = &unk_49DA100;
  v105 = 0x200000000LL;
  v120 = &unk_49DA0B0;
  v113 = 0;
  v114 = 0;
  v115 = 512;
  v117 = 0;
  v118 = 0;
  v107 = 0;
  v108 = 0;
  LOWORD(v109) = 0;
  sub_D5F1F0((__int64)&v104, v24 - 24);
  v28 = *(_QWORD *)(v24 - 120);
  if ( *(_BYTE *)v28 <= 0x1Cu )
    goto LABEL_38;
  v29 = *(_QWORD *)(v28 + 48);
  v102[0] = v29;
  if ( v29 && (sub_B96E90((__int64)v102, v29, 1), (v32 = v102[0]) != 0) )
  {
    v33 = (__int64)v104;
    v34 = v105;
    v35 = &v104[16 * (unsigned int)v105];
    if ( v104 != (_BYTE *)v35 )
    {
      while ( *(_DWORD *)v33 )
      {
        v33 += 16;
        if ( v35 == (_QWORD *)v33 )
          goto LABEL_69;
      }
      *(_QWORD *)(v33 + 8) = v102[0];
      goto LABEL_37;
    }
LABEL_69:
    if ( (unsigned int)v105 >= (unsigned __int64)HIDWORD(v105) )
    {
      v79 = (unsigned int)v105 + 1LL;
      if ( HIDWORD(v105) < v79 )
      {
        sub_C8D5F0((__int64)&v104, v106, v79, 0x10u, v30, v31);
        v35 = &v104[16 * (unsigned int)v105];
      }
      *v35 = 0;
      v35[1] = v32;
      v32 = v102[0];
      LODWORD(v105) = v105 + 1;
    }
    else
    {
      if ( v35 )
      {
        *(_DWORD *)v35 = 0;
        v35[1] = v32;
        v34 = v105;
        v32 = v102[0];
      }
      LODWORD(v105) = v34 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v104, 0);
    v32 = v102[0];
  }
  if ( v32 )
LABEL_37:
    sub_B91220((__int64)v102, v32);
LABEL_38:
  v36 = sub_D97050(*(_QWORD *)(a1 + 8), *(_QWORD *)(v95 + 8));
  if ( v36 <= (unsigned int)sub_D97050(*(_QWORD *)(a1 + 8), *(_QWORD *)(v23 + 8)) )
    goto LABEL_39;
  v46 = sub_DD8400(*(_QWORD *)(a1 + 8), v95);
  v47 = sub_DC5200(*(_QWORD *)(a1 + 8), (__int64)v46, *(_QWORD *)(v23 + 8), 0);
  if ( v46 == sub_DC2B70(*(_QWORD *)(a1 + 8), (__int64)v47, *(_QWORD *)(v95 + 8), 0) )
  {
    v101 = 1;
    v72 = *(_QWORD *)(a5 + 8);
    v99[0] = "wide.trip.count";
    v100 = 3;
    if ( v72 == *(_QWORD *)(v23 + 8) )
    {
      v73 = v23;
    }
    else
    {
      v73 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v111 + 15))(v111, 39, v23, v72);
      if ( !v73 )
      {
        v103 = 257;
        v80 = sub_BD2C40(72, unk_3F10A14);
        v73 = (__int64)v80;
        if ( v80 )
          sub_B515B0((__int64)v80, v23, v72, (__int64)v102, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v112 + 2))(v112, v73, v99, v108, v109);
        v81 = &v104[16 * (unsigned int)v105];
        if ( v104 != v81 )
        {
          v86 = v24;
          v82 = (__int64)v104;
          do
          {
            v83 = *(_QWORD *)(v82 + 8);
            v84 = *(_DWORD *)v82;
            v82 += 16;
            sub_B99FD0(v73, v84, v83);
          }
          while ( v81 != (_BYTE *)v82 );
          v24 = v86;
        }
      }
    }
    v23 = v73;
    goto LABEL_97;
  }
  if ( v46 == sub_DC5000(*(_QWORD *)(a1 + 8), (__int64)v47, *(_QWORD *)(v95 + 8), 0) )
  {
    v67 = *(_QWORD *)(a5 + 8);
    v102[0] = (unsigned __int64)"wide.trip.count";
    v103 = 259;
    v23 = sub_10A0620((__int64 *)&v104, v23, v67, (__int64)v102);
LABEL_97:
    sub_D4B3B0(a2, (unsigned __int8 *)v23, v102, 0, 0, 0);
    goto LABEL_39;
  }
  v101 = 1;
  v99[0] = "lftr.wideiv";
  v100 = 3;
  v48 = *(_QWORD *)(v23 + 8);
  if ( v48 == *(_QWORD *)(v95 + 8) )
  {
    v49 = v95;
  }
  else
  {
    v49 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD))*v111 + 15))(
            v111,
            38,
            v95,
            *(_QWORD *)(v23 + 8));
    if ( !v49 )
    {
      v103 = 257;
      v96 = sub_B51D30(38, v95, v48, (__int64)v102, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v112 + 2))(v112, v96, v99, v108, v109);
      v49 = v96;
      if ( v104 != &v104[16 * (unsigned int)v105] )
      {
        v97 = v24;
        v74 = (__int64)v104;
        v75 = &v104[16 * (unsigned int)v105];
        v76 = v49;
        do
        {
          v77 = *(_QWORD *)(v74 + 8);
          v78 = *(_DWORD *)v74;
          v74 += 16;
          sub_B99FD0(v76, v78, v77);
        }
        while ( v75 != (_BYTE *)v74 );
        v24 = v97;
        v49 = v76;
      }
    }
  }
  v95 = v49;
LABEL_39:
  v101 = 1;
  v99[0] = "exitcond";
  v100 = 3;
  v38 = (_QWORD *)(*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v111 + 7))(v111, v93, v95, v23);
  if ( !v38 )
  {
    v103 = 257;
    v38 = sub_BD2C40(72, unk_3F10FD0);
    if ( v38 )
    {
      v55 = *(_QWORD ***)(v95 + 8);
      v56 = *((unsigned __int8 *)v55 + 8);
      if ( (unsigned int)(v56 - 17) > 1 )
      {
        v58 = sub_BCB2A0(*v55);
      }
      else
      {
        BYTE4(v98) = (_BYTE)v56 == 18;
        LODWORD(v98) = *((_DWORD *)v55 + 8);
        v57 = (__int64 *)sub_BCB2A0(*v55);
        v58 = sub_BCE1B0(v57, v98);
      }
      sub_B523C0((__int64)v38, v58, 53, v93, v95, v23, (__int64)v102, 0, 0, 0);
    }
    (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, __int64, __int64))*v112 + 2))(v112, v38, v99, v108, v109);
    v59 = (__int64)v104;
    v60 = &v104[16 * (unsigned int)v105];
    if ( v104 != v60 )
    {
      do
      {
        v61 = *(_QWORD *)(v59 + 8);
        v62 = *(_DWORD *)v59;
        v59 += 16;
        sub_B99FD0((__int64)v38, v62, v61);
      }
      while ( v60 != (_BYTE *)v59 );
    }
    v39 = *(_QWORD *)(v24 - 120);
    if ( !v39 || (v40 = *(_QWORD *)(v24 - 112), (**(_QWORD **)(v24 - 104) = v40) == 0) )
    {
LABEL_43:
      *(_QWORD *)(v24 - 120) = v38;
      if ( !v38 )
        goto LABEL_47;
      goto LABEL_44;
    }
LABEL_42:
    *(_QWORD *)(v40 + 16) = *(_QWORD *)(v24 - 104);
    goto LABEL_43;
  }
  v39 = *(_QWORD *)(v24 - 120);
  if ( v39 )
  {
    v40 = *(_QWORD *)(v24 - 112);
    **(_QWORD **)(v24 - 104) = v40;
    if ( v40 )
      goto LABEL_42;
  }
  *(_QWORD *)(v24 - 120) = v38;
LABEL_44:
  v41 = v38[2];
  *(_QWORD *)(v24 - 112) = v41;
  if ( v41 )
    *(_QWORD *)(v41 + 16) = v24 - 112;
  *(_QWORD *)(v24 - 104) = v38 + 2;
  v38[2] = v24 - 120;
LABEL_47:
  v42 = *(unsigned int *)(a1 + 64);
  v43 = v42;
  if ( *(_DWORD *)(a1 + 68) <= (unsigned int)v42 )
  {
    v68 = (unsigned __int64 *)sub_C8D7D0(a1 + 56, a1 + 72, 0, 0x18u, v102, v37);
    v69 = &v68[3 * *(unsigned int *)(a1 + 64)];
    if ( v69 )
    {
      *v69 = 6;
      v69[1] = 0;
      v69[2] = v39;
      if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
        sub_BD73F0((__int64)v69);
    }
    sub_F17F80(a1 + 56, v68);
    v70 = *(_QWORD *)(a1 + 56);
    v71 = v102[0];
    if ( a1 + 72 != v70 )
      _libc_free(v70);
    ++*(_DWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = v68;
    *(_DWORD *)(a1 + 68) = v71;
  }
  else
  {
    v44 = (_QWORD *)(*(_QWORD *)(a1 + 56) + 24 * v42);
    if ( v44 )
    {
      *v44 = 6;
      v44[1] = 0;
      v44[2] = v39;
      if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
        sub_BD73F0((__int64)v44);
      v43 = *(_DWORD *)(a1 + 64);
    }
    *(_DWORD *)(a1 + 64) = v43 + 1;
  }
  nullsub_61();
  v119 = &unk_49DA100;
  nullsub_63();
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  return 1;
}
