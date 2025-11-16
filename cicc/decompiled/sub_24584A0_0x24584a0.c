// Function: sub_24584A0
// Address: 0x24584a0
//
__int64 __fastcall sub_24584A0(_QWORD ***a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 (__fastcall *v11)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v12; // rax
  _BYTE **v13; // rcx
  __int64 v14; // r12
  _QWORD *v15; // r14
  _QWORD *v16; // r13
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v27; // r10
  unsigned int *v28; // r14
  unsigned int *v29; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // rbx
  _QWORD **v35; // rdi
  int v36; // r11d
  _QWORD *v37; // r14
  unsigned int v38; // ecx
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _BYTE *v41; // r15
  _BYTE *v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  int v45; // edx
  int v46; // edx
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rax
  _BYTE *v52; // rax
  __int64 v53; // rbx
  __int64 v54; // rax
  unsigned __int8 v55; // al
  _QWORD *v56; // rax
  __int64 v57; // r9
  __int64 v58; // r14
  __int64 v59; // rbx
  unsigned __int64 *v60; // rbx
  unsigned __int64 v61; // r12
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // rax
  int v65; // r10d
  int v66; // r10d
  _QWORD **v67; // r8
  unsigned int v68; // edx
  __int64 v69; // rdi
  int v70; // esi
  _QWORD *v71; // rcx
  int v72; // r9d
  int v73; // r9d
  _QWORD **v74; // rdi
  int v75; // ecx
  _QWORD *v76; // rdx
  unsigned int v77; // r13d
  __int64 v78; // rsi
  unsigned int v79; // [rsp+0h] [rbp-220h]
  __int64 *v80; // [rsp+28h] [rbp-1F8h]
  char v81; // [rsp+30h] [rbp-1F0h]
  __int64 **v82; // [rsp+30h] [rbp-1F0h]
  __int64 v83; // [rsp+40h] [rbp-1E0h]
  __int64 v84; // [rsp+68h] [rbp-1B8h]
  const char *v85; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v86; // [rsp+78h] [rbp-1A8h]
  _BYTE v87[32]; // [rsp+80h] [rbp-1A0h] BYREF
  int v88[8]; // [rsp+A0h] [rbp-180h] BYREF
  __int16 v89; // [rsp+C0h] [rbp-160h]
  unsigned int *v90; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-148h]
  _BYTE v92[32]; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v93; // [rsp+100h] [rbp-120h]
  __int64 v94; // [rsp+108h] [rbp-118h]
  __int64 v95; // [rsp+110h] [rbp-110h]
  _QWORD *v96; // [rsp+118h] [rbp-108h]
  void **v97; // [rsp+120h] [rbp-100h]
  void **v98; // [rsp+128h] [rbp-F8h]
  __int64 v99; // [rsp+130h] [rbp-F0h]
  int v100; // [rsp+138h] [rbp-E8h]
  __int16 v101; // [rsp+13Ch] [rbp-E4h]
  char v102; // [rsp+13Eh] [rbp-E2h]
  __int64 v103; // [rsp+140h] [rbp-E0h]
  __int64 v104; // [rsp+148h] [rbp-D8h]
  void *v105; // [rsp+150h] [rbp-D0h] BYREF
  void *v106; // [rsp+158h] [rbp-C8h] BYREF
  unsigned __int64 *v107; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+168h] [rbp-B8h]
  unsigned __int64 v109; // [rsp+170h] [rbp-B0h] BYREF
  unsigned int v110; // [rsp+178h] [rbp-A8h]
  __int16 v111; // [rsp+180h] [rbp-A0h]
  __int64 v112; // [rsp+190h] [rbp-90h]
  __int64 v113; // [rsp+198h] [rbp-88h]
  __int64 v114; // [rsp+1A0h] [rbp-80h]
  __int64 v115; // [rsp+1A8h] [rbp-78h]
  void **v116; // [rsp+1B0h] [rbp-70h]
  void **v117; // [rsp+1B8h] [rbp-68h]
  __int64 v118; // [rsp+1C0h] [rbp-60h]
  int v119; // [rsp+1C8h] [rbp-58h]
  __int16 v120; // [rsp+1CCh] [rbp-54h]
  char v121; // [rsp+1CEh] [rbp-52h]
  __int64 v122; // [rsp+1D0h] [rbp-50h]
  __int64 v123; // [rsp+1D8h] [rbp-48h]
  void *v124; // [rsp+1E0h] [rbp-40h] BYREF
  void *v125; // [rsp+1E8h] [rbp-38h] BYREF

  v4 = sub_2453E10((__int64)a1, a2);
  v96 = (_QWORD *)sub_BD5C60(a2);
  v97 = &v105;
  v98 = &v106;
  v101 = 512;
  v105 = &unk_49DA100;
  v90 = (unsigned int *)v92;
  v106 = &unk_49DA0B0;
  v91 = 0x200000000LL;
  v99 = 0;
  v100 = 0;
  v102 = 7;
  v103 = 0;
  v104 = 0;
  v93 = 0;
  v94 = 0;
  LOWORD(v95) = 0;
  sub_D5F1F0((__int64)&v90, a2);
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 || (v81 = *(_BYTE *)v5) != 0 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  if ( *(_DWORD *)(v5 + 36) == 202 )
    sub_B2F770(v4, 3u);
  v89 = 257;
  v6 = sub_B59BC0(a2);
  if ( *(_DWORD *)(v6 + 32) <= 0x40u )
    v7 = *(_QWORD *)(v6 + 24);
  else
    v7 = **(_QWORD **)(v6 + 24);
  v8 = *(_QWORD *)(v4 + 24);
  v9 = sub_BCB2D0(v96);
  v85 = (const char *)sub_ACD640(v9, 0, 0);
  v10 = sub_BCB2D0(v96);
  v86 = sub_ACD640(v10, (unsigned int)v7, 0);
  v11 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v97 + 8);
  if ( v11 == sub_920540 )
  {
    if ( sub_BCEA30(v8) )
      goto LABEL_36;
    if ( *(_BYTE *)v4 > 0x15u )
      goto LABEL_36;
    v12 = sub_2450470((_BYTE **)&v85, (__int64)v87);
    if ( v13 != v12 )
      goto LABEL_36;
    LOBYTE(v111) = 0;
    v14 = sub_AD9FD0(v8, (unsigned __int8 *)v4, (__int64 *)&v85, 2, 3u, (__int64)&v107, 0);
    if ( (_BYTE)v111 )
    {
      LOBYTE(v111) = 0;
      if ( v110 > 0x40 && v109 )
        j_j___libc_free_0_0(v109);
      if ( (unsigned int)v108 > 0x40 && v107 )
        j_j___libc_free_0_0((unsigned __int64)v107);
    }
  }
  else
  {
    v14 = v11((__int64)v97, v8, (_BYTE *)v4, (_BYTE **)&v85, 2, 3);
  }
  if ( v14 )
    goto LABEL_14;
LABEL_36:
  v111 = 257;
  v14 = (__int64)sub_BD2C40(88, 3u);
  if ( !v14 )
    goto LABEL_39;
  v27 = *(_QWORD *)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 > 1 )
  {
    v44 = *((_QWORD *)v85 + 1);
    v45 = *(unsigned __int8 *)(v44 + 8);
    if ( v45 != 17 )
    {
      if ( v45 == 18 )
      {
LABEL_53:
        v81 = 1;
        goto LABEL_54;
      }
      v44 = *(_QWORD *)(v86 + 8);
      v46 = *(unsigned __int8 *)(v44 + 8);
      if ( v46 != 17 )
      {
        if ( v46 != 18 )
          goto LABEL_38;
        goto LABEL_53;
      }
    }
LABEL_54:
    LODWORD(v84) = *(_DWORD *)(v44 + 32);
    BYTE4(v84) = v81;
    v27 = sub_BCE1B0((__int64 *)v27, v84);
  }
LABEL_38:
  sub_B44260(v14, v27, 34, 3u, 0, 0);
  *(_QWORD *)(v14 + 72) = v8;
  *(_QWORD *)(v14 + 80) = sub_B4DC50(v8, (__int64)&v85, 2);
  sub_B4D9A0(v14, v4, (__int64 *)&v85, 2, (__int64)&v107);
LABEL_39:
  sub_B4DDE0(v14, 3);
  (*((void (__fastcall **)(void **, __int64, int *, __int64, __int64))*v98 + 2))(v98, v14, v88, v94, v95);
  v28 = v90;
  v29 = &v90[4 * (unsigned int)v91];
  if ( v90 != v29 )
  {
    do
    {
      v30 = *((_QWORD *)v28 + 1);
      v31 = *v28;
      v28 += 4;
      sub_B99FD0(v14, v31, v30);
    }
    while ( v29 != v28 );
  }
LABEL_14:
  if ( *((_DWORD *)a1 + 25) == 5 )
    goto LABEL_33;
  v15 = sub_C52410();
  v16 = v15 + 1;
  v17 = sub_C959E0();
  v18 = (_QWORD *)v15[2];
  if ( v18 )
  {
    v19 = v15 + 1;
    do
    {
      while ( 1 )
      {
        v20 = v18[2];
        v21 = v18[3];
        if ( v17 <= v18[4] )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v21 )
          goto LABEL_20;
      }
      v19 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v20 );
LABEL_20:
    if ( v16 != v19 && v17 >= v19[4] )
      v16 = v19;
  }
  if ( v16 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_42;
  v22 = v16[7];
  if ( !v22 )
    goto LABEL_42;
  v23 = v16 + 6;
  do
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(v22 + 16);
      v25 = *(_QWORD *)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) >= dword_4FE7228 )
        break;
      v22 = *(_QWORD *)(v22 + 24);
      if ( !v25 )
        goto LABEL_29;
    }
    v23 = (_QWORD *)v22;
    v22 = *(_QWORD *)(v22 + 16);
  }
  while ( v24 );
LABEL_29:
  if ( v16 + 6 == v23 || dword_4FE7228 < *((_DWORD *)v23 + 8) || *((int *)v23 + 9) <= 0 )
  {
LABEL_42:
    if ( *((_DWORD *)a1 + 23) != 4 )
      goto LABEL_33;
  }
  else if ( !byte_4FE72A8 )
  {
    goto LABEL_33;
  }
  v32 = sub_BCB2E0(**a1);
  v33 = *((_DWORD *)a1 + 60);
  v82 = (__int64 **)v32;
  v34 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
  if ( !v33 )
  {
    a1[27] = (_QWORD **)((char *)a1[27] + 1);
    goto LABEL_86;
  }
  v35 = a1[28];
  v36 = 1;
  v37 = 0;
  v38 = (v33 - 1) & (((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9));
  v39 = &v35[2 * v38];
  v40 = *v39;
  if ( v34 != *v39 )
  {
    while ( v40 != -4096 )
    {
      if ( v40 == -8192 && !v37 )
        v37 = v39;
      v38 = (v33 - 1) & (v36 + v38);
      v39 = &v35[2 * v38];
      v40 = *v39;
      if ( v34 == *v39 )
        goto LABEL_45;
      ++v36;
    }
    if ( !v37 )
      v37 = v39;
    v47 = *((_DWORD *)a1 + 58);
    a1[27] = (_QWORD **)((char *)a1[27] + 1);
    v48 = v47 + 1;
    if ( 4 * v48 < 3 * v33 )
    {
      if ( v33 - *((_DWORD *)a1 + 59) - v48 > v33 >> 3 )
      {
LABEL_71:
        *((_DWORD *)a1 + 58) = v48;
        if ( *v37 != -4096 )
          --*((_DWORD *)a1 + 59);
        *v37 = v34;
        v37[1] = 0;
        v80 = v37 + 1;
        goto LABEL_74;
      }
      sub_24582C0((__int64)(a1 + 27), v33);
      v72 = *((_DWORD *)a1 + 60);
      if ( v72 )
      {
        v73 = v72 - 1;
        v74 = a1[28];
        v75 = 1;
        v76 = 0;
        v77 = v73 & (((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9));
        v37 = &v74[2 * v77];
        v78 = *v37;
        v48 = *((_DWORD *)a1 + 58) + 1;
        if ( v34 != *v37 )
        {
          while ( v78 != -4096 )
          {
            if ( !v76 && v78 == -8192 )
              v76 = v37;
            v77 = v73 & (v75 + v77);
            v37 = &v74[2 * v77];
            v78 = *v37;
            if ( v34 == *v37 )
              goto LABEL_71;
            ++v75;
          }
          if ( v76 )
            v37 = v76;
        }
        goto LABEL_71;
      }
LABEL_111:
      ++*((_DWORD *)a1 + 58);
      BUG();
    }
LABEL_86:
    sub_24582C0((__int64)(a1 + 27), 2 * v33);
    v65 = *((_DWORD *)a1 + 60);
    if ( v65 )
    {
      v66 = v65 - 1;
      v67 = a1[28];
      v68 = v66 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v37 = &v67[2 * v68];
      v69 = *v37;
      v48 = *((_DWORD *)a1 + 58) + 1;
      if ( v34 != *v37 )
      {
        v70 = 1;
        v71 = 0;
        while ( v69 != -4096 )
        {
          if ( !v71 && v69 == -8192 )
            v71 = v37;
          v68 = v66 & (v70 + v68);
          v37 = &v67[2 * v68];
          v69 = *v37;
          if ( v34 == *v37 )
            goto LABEL_71;
          ++v70;
        }
        if ( v71 )
          v37 = v71;
      }
      goto LABEL_71;
    }
    goto LABEL_111;
  }
LABEL_45:
  v80 = v39 + 1;
  if ( !v39[1] )
  {
LABEL_74:
    v49 = *(_QWORD *)(v34 + 80);
    if ( !v49 )
      BUG();
    v50 = *(_QWORD *)(v49 + 32);
    if ( v50 )
      v50 -= 24;
    v51 = sub_BD5C60(v50);
    v121 = 7;
    v115 = v51;
    v116 = &v124;
    v117 = &v125;
    v107 = &v109;
    v124 = &unk_49DA100;
    v120 = 512;
    LOWORD(v114) = 0;
    v108 = 0x200000000LL;
    v125 = &unk_49DA0B0;
    v118 = 0;
    v119 = 0;
    v122 = 0;
    v123 = 0;
    v112 = 0;
    v113 = 0;
    sub_D5F1F0((__int64)&v107, v50);
    v52 = sub_2450D10(a1, "__llvm_profile_counter_bias", 0x1Bu);
    v87[17] = 1;
    v53 = (__int64)v52;
    v87[16] = 3;
    v85 = "profc_bias";
    v54 = sub_AA4E30(v112);
    v55 = sub_AE5020(v54, (__int64)v82);
    v89 = 257;
    v79 = v55;
    v56 = sub_BD2C40(80, unk_3F10A14);
    v57 = v79;
    v58 = (__int64)v56;
    if ( v56 )
      sub_B4D190((__int64)v56, (__int64)v82, v53, (__int64)v88, 0, v79, 0, 0);
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64, __int64))*v117 + 2))(
      v117,
      v58,
      &v85,
      v113,
      v114,
      v57);
    v59 = 2LL * (unsigned int)v108;
    if ( v107 != &v107[v59] )
    {
      v83 = v14;
      v60 = &v107[v59];
      v61 = (unsigned __int64)v107;
      do
      {
        v62 = *(_QWORD *)(v61 + 8);
        v63 = *(_DWORD *)v61;
        v61 += 16LL;
        sub_B99FD0(v58, v63, v62);
      }
      while ( v60 != (unsigned __int64 *)v61 );
      v14 = v83;
    }
    *v80 = v58;
    v64 = sub_B9C770(**a1, 0, 0, 0, 1);
    sub_B99FD0(v58, 6u, v64);
    nullsub_61();
    v124 = &unk_49DA100;
    nullsub_63();
    if ( v107 != &v109 )
      _libc_free((unsigned __int64)v107);
  }
  v111 = 257;
  v41 = (_BYTE *)*v80;
  v89 = 257;
  v42 = (_BYTE *)sub_2452130((__int64 *)&v90, 0x2Fu, v14, v82, (__int64)v88, 0, (int)v85, 0);
  v43 = sub_929C50(&v90, v42, v41, (__int64)&v107, 0, 0);
  v111 = 257;
  v14 = sub_2452130((__int64 *)&v90, 0x30u, v43, *(__int64 ***)(v14 + 8), (__int64)&v107, 0, v88[0], 0);
LABEL_33:
  nullsub_61();
  v105 = &unk_49DA100;
  nullsub_63();
  if ( v90 != (unsigned int *)v92 )
    _libc_free((unsigned __int64)v90);
  return v14;
}
