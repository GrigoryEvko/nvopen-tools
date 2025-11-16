// Function: sub_328A0F0
// Address: 0x328a0f0
//
__int64 __fastcall sub_328A0F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  _DWORD *v10; // rax
  __int64 v11; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r11
  unsigned __int16 *v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  bool v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v28; // rax
  unsigned int v29; // edx
  char **v30; // rsi
  unsigned __int64 v31; // rdx
  unsigned int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // rax
  unsigned int v35; // edx
  unsigned __int16 *v36; // rdx
  int v37; // eax
  __int64 v38; // rdx
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  _QWORD *v41; // rax
  unsigned int v42; // r15d
  char v43; // al
  int v44; // r9d
  unsigned int v45; // r10d
  __int64 v46; // r11
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rdx
  _QWORD *v54; // rax
  __int64 v55; // rdx
  unsigned int *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  unsigned __int64 v59; // r8
  __int64 v60; // rdi
  int v61; // ecx
  unsigned __int64 v62; // r8
  unsigned __int64 v63; // rcx
  int v64; // eax
  bool v65; // al
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // rax
  unsigned __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rax
  int v72; // eax
  __int64 v73; // rdx
  int v74; // eax
  unsigned __int64 v75; // rax
  bool v76; // al
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int16 v80; // ax
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int16 v84; // ax
  __int64 v85; // rdx
  __int64 v86; // [rsp-10h] [rbp-100h]
  unsigned int v87; // [rsp+Ch] [rbp-E4h]
  __int64 v88; // [rsp+18h] [rbp-D8h]
  __int64 v89; // [rsp+20h] [rbp-D0h]
  __int128 v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+20h] [rbp-D0h]
  __int64 v92; // [rsp+30h] [rbp-C0h]
  unsigned int v93; // [rsp+30h] [rbp-C0h]
  __int64 v94; // [rsp+30h] [rbp-C0h]
  int v95; // [rsp+30h] [rbp-C0h]
  unsigned int v96; // [rsp+30h] [rbp-C0h]
  __int64 v97; // [rsp+38h] [rbp-B8h]
  unsigned int v98; // [rsp+38h] [rbp-B8h]
  unsigned int v99; // [rsp+40h] [rbp-B0h]
  __int64 v100; // [rsp+40h] [rbp-B0h]
  __int64 v101; // [rsp+48h] [rbp-A8h]
  __int64 v102; // [rsp+48h] [rbp-A8h]
  __int64 v103; // [rsp+48h] [rbp-A8h]
  __int64 v104; // [rsp+48h] [rbp-A8h]
  __int64 v105; // [rsp+48h] [rbp-A8h]
  __int64 v106; // [rsp+48h] [rbp-A8h]
  __int64 v107; // [rsp+48h] [rbp-A8h]
  __int64 v108; // [rsp+50h] [rbp-A0h]
  unsigned int v109; // [rsp+50h] [rbp-A0h]
  char **v110; // [rsp+50h] [rbp-A0h]
  __int64 v111; // [rsp+50h] [rbp-A0h]
  __int64 v112; // [rsp+50h] [rbp-A0h]
  __int16 v113; // [rsp+50h] [rbp-A0h]
  int v114; // [rsp+58h] [rbp-98h]
  int v115; // [rsp+58h] [rbp-98h]
  __int64 v116; // [rsp+58h] [rbp-98h]
  unsigned int v117; // [rsp+70h] [rbp-80h] BYREF
  __int64 v118; // [rsp+78h] [rbp-78h]
  __int64 v119; // [rsp+80h] [rbp-70h] BYREF
  __int64 v120; // [rsp+88h] [rbp-68h]
  __int64 v121; // [rsp+90h] [rbp-60h]
  __int64 v122; // [rsp+98h] [rbp-58h]
  int v123; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v124; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v125; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v126; // [rsp+B8h] [rbp-38h]

  v7 = a1;
  v8 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL), 0, 0);
  if ( !v8 )
    return 0;
  v9 = v8;
  v10 = *(_DWORD **)(a1 + 40);
  v11 = *(_QWORD *)v10;
  if ( *(_DWORD *)(*(_QWORD *)v10 + 24LL) != 58 )
    return 0;
  v13 = v10[2];
  v14 = *(__int64 **)(v11 + 40);
  v15 = *v14;
  v99 = *((_DWORD *)v14 + 2);
  v108 = v14[5];
  v101 = v14[6];
  v114 = *(_DWORD *)(*v14 + 24);
  if ( (unsigned int)(v114 - 213) > 1 )
    return 0;
  v88 = v14[5];
  v16 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v15 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v15 + 40) + 8LL));
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v117) = v17;
  v118 = v18;
  if ( (_WORD)v17 )
  {
    if ( (unsigned __int16)(v17 - 17) > 0xD3u )
    {
      LOWORD(v125) = v17;
      v126 = v18;
      goto LABEL_19;
    }
    LOWORD(v17) = word_4456580[v17 - 1];
    v58 = 0;
  }
  else
  {
    v87 = v13;
    v92 = v18;
    v89 = v15;
    v19 = sub_30070B0((__int64)&v117);
    v15 = v89;
    v13 = v87;
    if ( !v19 )
    {
      v126 = v92;
      LOWORD(v125) = 0;
LABEL_8:
      v93 = v13;
      v97 = v15;
      v20 = sub_3007260((__int64)&v125);
      v13 = v93;
      v121 = v20;
      v15 = v97;
      v122 = v21;
      goto LABEL_9;
    }
    LOWORD(v17) = sub_3009970((__int64)&v117, v11, v92, v87, (__int64)&v117);
    v13 = v87;
    v15 = v89;
  }
  LOWORD(v125) = v17;
  v126 = v58;
  if ( !(_WORD)v17 )
    goto LABEL_8;
LABEL_19:
  if ( (_WORD)v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
LABEL_103:
    BUG();
  v20 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v17 - 16];
LABEL_9:
  v22 = *(_QWORD *)(v11 + 56);
  v98 = v20;
  if ( !v22 )
    goto LABEL_49;
  v23 = *(_QWORD *)(v11 + 56);
  v24 = 1;
  do
  {
    if ( *(_DWORD *)(v23 + 8) == v13 )
    {
      if ( !v24 )
        goto LABEL_49;
      v23 = *(_QWORD *)(v23 + 32);
      if ( !v23 )
        goto LABEL_23;
      if ( v13 == *(_DWORD *)(v23 + 8) )
        goto LABEL_49;
      v24 = 0;
    }
    v23 = *(_QWORD *)(v23 + 32);
  }
  while ( v23 );
  if ( v24 == 1 )
  {
LABEL_49:
    v50 = 1;
    if ( (_WORD)v117 == 1
      || (_WORD)v117 && (v50 = (unsigned __int16)v117, *(_QWORD *)(a4 + 8LL * (unsigned __int16)v117 + 112)) )
    {
      if ( (*(_BYTE *)((unsigned int)(v114 != 213) + 63 + a4 + 500 * v50 + 6414) & 0xFB) == 0 && v22 )
      {
        v51 = *(_QWORD *)(v11 + 56);
        v91 = v15;
        do
        {
          v55 = *(_QWORD *)(v51 + 16);
          if ( (unsigned int)(*(_DWORD *)(v55 + 24) - 191) > 1 )
            return 0;
          v52 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v55 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v55 + 40) + 48LL), 0, 0);
          if ( !v52 )
            return 0;
          v53 = *(_QWORD *)(v52 + 96);
          v54 = *(_QWORD **)(v53 + 24);
          if ( *(_DWORD *)(v53 + 32) > 0x40u )
            v54 = (_QWORD *)*v54;
          if ( v98 > (unsigned int)v54 )
            return 0;
          v51 = *(_QWORD *)(v51 + 32);
        }
        while ( v51 );
        v15 = v91;
        v7 = a1;
      }
    }
  }
LABEL_23:
  v25 = v101;
  v94 = v15;
  v26 = sub_33DFBC0(v108, v101, 0, 0);
  v27 = v94;
  if ( v26 )
  {
    v28 = *(_QWORD *)(v26 + 96);
    v29 = *(_DWORD *)(v28 + 32);
    v30 = (char **)(v28 + 24);
    if ( v114 == 213 )
    {
      v59 = *(_QWORD *)(v28 + 24);
      v60 = 1LL << ((unsigned __int8)v29 - 1);
      if ( v29 > 0x40 )
      {
        v96 = *(_DWORD *)(v28 + 32);
        v106 = v27;
        if ( (*(_QWORD *)(v59 + 8LL * ((v29 - 1) >> 6)) & v60) != 0 )
          v72 = sub_C44500((__int64)v30);
        else
          v72 = sub_C444A0((__int64)v30);
        v27 = v106;
        v29 = v96;
        v61 = v72;
      }
      else if ( (v59 & v60) != 0 )
      {
        if ( v29 )
        {
          v61 = 64;
          v62 = ~(v59 << (64 - (unsigned __int8)v29));
          if ( v62 )
          {
            _BitScanReverse64(&v63, v62);
            v61 = v63 ^ 0x3F;
          }
        }
        else
        {
          v61 = 0;
        }
      }
      else
      {
        v74 = 64;
        if ( v59 )
        {
          _BitScanReverse64(&v75, v59);
          v74 = v75 ^ 0x3F;
        }
        v61 = v29 + v74 - 64;
      }
      v32 = v29 + 1 - v61;
    }
    else if ( v29 > 0x40 )
    {
      v95 = *(_DWORD *)(v28 + 32);
      v104 = v27;
      v110 = (char **)(v28 + 24);
      v64 = sub_C444A0(v28 + 24);
      v27 = v104;
      v30 = v110;
      v32 = v95 - v64;
    }
    else
    {
      v31 = *(_QWORD *)(v28 + 24);
      if ( !v31 )
      {
LABEL_29:
        v102 = v27;
        v33 = sub_32844A0((unsigned __int16 *)&v117, (__int64)v30);
        sub_C44740((__int64)&v125, v30, v33);
        v34 = sub_34007B0(a3, (unsigned int)&v125, a2, v117, v118, 0, 0);
        v25 = v86;
        v27 = v102;
        *(_QWORD *)&v90 = v34;
        *((_QWORD *)&v90 + 1) = v35;
        if ( (unsigned int)v126 > 0x40 && v125 )
        {
          j_j___libc_free_0_0(v125);
          v27 = v102;
        }
        goto LABEL_32;
      }
      _BitScanReverse64(&v31, v31);
      v32 = 64 - (v31 ^ 0x3F);
    }
    if ( v32 > v98 )
      return 0;
    goto LABEL_29;
  }
  if ( *(_DWORD *)(v88 + 24) != *(_DWORD *)(v94 + 24) )
    return 0;
  v56 = *(unsigned int **)(v88 + 40);
  v57 = *(_QWORD *)(*(_QWORD *)v56 + 48LL) + 16LL * v56[2];
  if ( (_WORD)v117 != *(_WORD *)v57 || !(_WORD)v117 && v118 != *(_QWORD *)(v57 + 8) )
    return 0;
  *(_QWORD *)&v90 = *(_QWORD *)v56;
  *((_QWORD *)&v90 + 1) = v56[2];
LABEL_32:
  v36 = (unsigned __int16 *)(*(_QWORD *)(v27 + 48) + 16LL * v99);
  v37 = *v36;
  v38 = *((_QWORD *)v36 + 1);
  LOWORD(v119) = v37;
  v120 = v38;
  if ( !(_WORD)v37 )
  {
    v100 = v38;
    v105 = v27;
    v65 = sub_30070B0((__int64)&v119);
    v27 = v105;
    if ( !v65 )
    {
      v124 = v100;
      LOWORD(v123) = 0;
      goto LABEL_79;
    }
    LOWORD(v37) = sub_3009970((__int64)&v119, v25, v100, v66, v67);
    v27 = v105;
LABEL_85:
    LOWORD(v123) = v37;
    v124 = v73;
    if ( (_WORD)v37 )
      goto LABEL_35;
LABEL_79:
    v111 = v27;
    v68 = sub_3007260((__int64)&v123);
    v27 = v111;
    v69 = v68;
    v71 = v70;
    v125 = v69;
    v39 = v69;
    v126 = v71;
    goto LABEL_38;
  }
  if ( (unsigned __int16)(v37 - 17) <= 0xD3u )
  {
    LOWORD(v37) = word_4456580[v37 - 1];
    v73 = 0;
    goto LABEL_85;
  }
  LOWORD(v123) = v37;
  v124 = v38;
LABEL_35:
  if ( (_WORD)v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
    goto LABEL_103;
  v39 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v37 - 16];
LABEL_38:
  if ( 2 * v98 != v39 )
    return 0;
  v40 = *(_QWORD *)(v9 + 96);
  v41 = *(_QWORD **)(v40 + 24);
  if ( *(_DWORD *)(v40 + 32) > 0x40u )
    v41 = (_QWORD *)*v41;
  if ( v98 != (_DWORD)v41 )
    return 0;
  v42 = (v114 == 213) + 172;
  if ( (_WORD)v117 )
  {
    if ( (unsigned __int16)(v117 - 17) > 0xD3u )
      goto LABEL_44;
    goto LABEL_95;
  }
  v112 = v27;
  v76 = sub_30070B0((__int64)&v117);
  v27 = v112;
  if ( v76 )
  {
LABEL_95:
    v77 = *(_QWORD *)(a3 + 64);
    v107 = v27;
    v123 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a4 + 592LL))(a4, v77, v117, v118);
    v124 = v78;
    v80 = sub_3281170(&v117, v77, v78, v79, (__int64)&v117);
    v116 = v81;
    v113 = v80;
    v84 = sub_3281170(&v123, v77, v81, v82, v83);
    if ( v84 == v113 && (v84 || v85 == v116) && (unsigned __int8)sub_328A020(a4, v42, v123, v124, 0) )
    {
      v45 = v117;
      v44 = v118;
      v46 = v107;
      goto LABEL_45;
    }
    return 0;
  }
LABEL_44:
  v103 = v27;
  v109 = v117;
  v115 = v118;
  v43 = sub_328A020(a4, v42, v117, v118, 0);
  v44 = v115;
  v45 = v109;
  v46 = v103;
  if ( !v43 )
    return 0;
LABEL_45:
  v48 = sub_3406EB0(a3, v42, a2, v45, v44, v44, *(_OWORD *)*(_QWORD *)(v46 + 40), v90);
  if ( *(_DWORD *)(v7 + 24) == 191 )
    return sub_33FB160(a3, v48, v47, a2, (unsigned int)v119, v120);
  else
    return sub_33FB310(a3, v48, v47, a2, (unsigned int)v119, v120);
}
