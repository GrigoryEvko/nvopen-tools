// Function: sub_17A3E70
// Address: 0x17a3e70
//
__int64 __fastcall sub_17A3E70(__int64 a1, _DWORD *a2, __int64 a3, int a4)
{
  __int64 v4; // r14
  unsigned int *v5; // r12
  __int64 v6; // r13
  _QWORD *v7; // rbx
  unsigned int v8; // ebx
  char v9; // cl
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  unsigned int v13; // edi
  __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  unsigned int v16; // ebx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int8 *v24; // r15
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r13
  int v28; // r13d
  __int64 v29; // rax
  int v30; // edi
  __int64 v31; // rax
  __int64 v32; // rbx
  int v33; // eax
  __int64 v34; // r14
  __int64 v35; // r12
  __int64 v36; // rbx
  __int64 v37; // rdi
  __int64 *v38; // r13
  int v39; // r9d
  int v40; // r13d
  bool v41; // sf
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rbx
  int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  int v49; // r13d
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // r8
  __int64 *v53; // r15
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // rsi
  unsigned __int8 *v58; // rsi
  __int64 v59; // rbx
  int v60; // r9d
  __int64 v61; // rdx
  int v62; // ebx
  int v63; // r8d
  unsigned int v64; // esi
  unsigned int v65; // r15d
  int v66; // eax
  unsigned __int64 v67; // rax
  char *v68; // rsi
  __int64 v69; // r15
  __int64 v70; // r12
  __int64 v71; // r14
  _BYTE *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 *v75; // r12
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 v82; // rsi
  unsigned __int8 *v83; // rax
  __int64 v84; // rdi
  unsigned __int64 *v85; // r14
  __int64 v86; // rax
  unsigned __int64 v87; // rcx
  __int64 v88; // rdx
  __int64 v89; // rsi
  __int64 v90; // rsi
  unsigned __int8 *v91; // rsi
  __int64 v92; // r15
  unsigned int v93; // eax
  unsigned int v94; // r12d
  __int64 v95; // rax
  unsigned int v98; // edx
  int v99; // [rsp-10h] [rbp-2B0h]
  __int64 v100; // [rsp+0h] [rbp-2A0h]
  unsigned int *v101; // [rsp+18h] [rbp-288h]
  __int64 v102; // [rsp+18h] [rbp-288h]
  int v103; // [rsp+20h] [rbp-280h]
  int v104; // [rsp+28h] [rbp-278h]
  int v105; // [rsp+2Ch] [rbp-274h]
  __int64 v106; // [rsp+30h] [rbp-270h]
  __int64 *v107; // [rsp+30h] [rbp-270h]
  __int64 v108; // [rsp+38h] [rbp-268h]
  int v109; // [rsp+38h] [rbp-268h]
  __int64 v110; // [rsp+40h] [rbp-260h]
  __int64 v111; // [rsp+48h] [rbp-258h]
  __int64 v112; // [rsp+48h] [rbp-258h]
  _QWORD *v114; // [rsp+50h] [rbp-250h]
  unsigned __int8 *v116; // [rsp+68h] [rbp-238h] BYREF
  unsigned int *v117[2]; // [rsp+70h] [rbp-230h] BYREF
  __int64 v118[2]; // [rsp+80h] [rbp-220h] BYREF
  __int16 v119; // [rsp+90h] [rbp-210h]
  char v120[16]; // [rsp+A0h] [rbp-200h] BYREF
  __int16 v121; // [rsp+B0h] [rbp-1F0h]
  __int64 *v122; // [rsp+C0h] [rbp-1E0h]
  __int64 v123; // [rsp+C8h] [rbp-1D8h]
  __int64 v124; // [rsp+D0h] [rbp-1D0h]
  unsigned __int8 *v125; // [rsp+D8h] [rbp-1C8h] BYREF
  unsigned __int8 *v126; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v127; // [rsp+E8h] [rbp-1B8h]
  _WORD v128[16]; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 *v129; // [rsp+110h] [rbp-190h] BYREF
  __int64 v130; // [rsp+118h] [rbp-188h]
  _BYTE v131[48]; // [rsp+120h] [rbp-180h] BYREF
  unsigned int *v132; // [rsp+150h] [rbp-150h] BYREF
  __int64 v133; // [rsp+158h] [rbp-148h]
  _BYTE v134[128]; // [rsp+160h] [rbp-140h] BYREF
  __int64 *v135; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v136; // [rsp+1E8h] [rbp-B8h]
  _BYTE v137[176]; // [rsp+1F0h] [rbp-B0h] BYREF

  v110 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  if ( (_DWORD)v110 == 1 )
    return 0;
  v4 = (__int64)a2;
  v5 = (unsigned int *)a3;
  if ( a4 >= 0 )
  {
    v6 = *(_QWORD *)&a2[6 * (a4 - (unsigned __int64)(a2[5] & 0xFFFFFFF))];
    if ( *(_BYTE *)(v6 + 16) == 13 )
    {
      v7 = *(_QWORD **)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
        v7 = (_QWORD *)*v7;
      v8 = (unsigned __int8)v7 & 0xF;
      v9 = sub_39FAC40(v8);
      v10 = *(_QWORD *)v5;
      v11 = (1 << v9) - 1;
      if ( v5[2] > 0x40 )
      {
        *(_QWORD *)v10 &= v11;
        memset((void *)(*(_QWORD *)v5 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)v5[2] + 63) >> 6) - 8);
      }
      else
      {
        *(_QWORD *)v5 = v10 & v11;
      }
      v12 = 0;
      v13 = 0;
      v14 = 0;
      do
      {
        if ( ((1 << v12) & v8) != 0 )
        {
          v15 = *(_QWORD *)v5;
          if ( v5[2] > 0x40 )
            v15 = *(_QWORD *)(v15 + 8LL * (v13 >> 6));
          if ( (v15 & (1LL << v13)) != 0 )
            v14 = (unsigned int)v14 | (1 << v12);
          ++v13;
        }
        ++v12;
      }
      while ( v12 != 4 );
      v108 = 0;
      if ( (_DWORD)v14 != v8 )
        v108 = sub_159C470(*(_QWORD *)v6, v14, 0);
      v16 = v5[2];
      if ( v16 > 0x40 )
        goto LABEL_19;
      goto LABEL_40;
    }
    return 0;
  }
  v16 = *(_DWORD *)(a3 + 8);
  if ( v16 > 0x40 )
  {
    **(_QWORD **)a3 = (1 << (v16 - sub_16A57B0(a3))) - 1;
    memset((void *)(*(_QWORD *)v5 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)v5[2] + 63) >> 6) - 8);
    v16 = v5[2];
    v108 = 0;
    if ( v16 > 0x40 )
    {
LABEL_19:
      v17 = (unsigned int)sub_16A5940((__int64)v5);
      goto LABEL_20;
    }
LABEL_40:
    v17 = (int)sub_39FAC40(*(_QWORD *)v5);
    if ( !(_DWORD)v17 )
      return sub_1599EF0(*(__int64 ***)v4);
    goto LABEL_21;
  }
  if ( !*(_QWORD *)a3 )
    return sub_1599EF0(*(__int64 ***)v4);
  _BitScanReverse64(&v26, *(_QWORD *)a3);
  v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ((1 << (64 - (v26 ^ 0x3F))) - 1);
  LODWORD(v17) = sub_39FAC40(v27);
  *(_QWORD *)v5 = v27;
  v108 = 0;
  v17 = (int)v17;
LABEL_20:
  if ( !v17 )
    return sub_1599EF0(*(__int64 ***)v4);
LABEL_21:
  v18 = (((v17 - 1) | ((unsigned __int64)(v17 - 1) >> 1)) >> 2) | (v17 - 1) | ((unsigned __int64)(v17 - 1) >> 1);
  v19 = (((v18 >> 4) | v18) >> 8) | (v18 >> 4) | v18;
  v111 = ((v19 >> 16) | v19) + 1;
  v104 = ((v19 >> 16) | v19) + 1;
  if ( ((unsigned int)(v19 >> 16) | (unsigned int)v19) == 0xFFFFFFFF )
    return sub_1599EF0(*(__int64 ***)v4);
  if ( (unsigned int)v111 >= (unsigned int)v110 )
  {
    if ( v16 > 0x40 )
    {
      v28 = sub_16A58F0((__int64)v5);
      if ( v28 && (unsigned int)sub_16A57B0((__int64)v5) + v28 == v16 )
      {
LABEL_26:
        if ( v108 )
        {
          v20 = (__int64 *)(v4 + 24 * ((unsigned int)a4 - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
          if ( *v20 )
          {
            v21 = v20[1];
            v22 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v22 = v21;
            if ( v21 )
              *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
          }
          *v20 = v108;
          v23 = *(_QWORD *)(v108 + 8);
          v20[1] = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v23 + 16) & 3LL;
          v20[2] = v20[2] & 3 | (v108 + 8);
          *(_QWORD *)(v108 + 8) = v20;
        }
        return 0;
      }
    }
    else if ( *(_QWORD *)v5 && (*(_QWORD *)v5 & (*(_QWORD *)v5 + 1LL)) == 0 )
    {
      goto LABEL_26;
    }
  }
  v29 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)(v29 + 16) )
    BUG();
  v30 = *(_DWORD *)(v29 + 36);
  v132 = (unsigned int *)v134;
  v133 = 0x1000000000LL;
  v103 = v30;
  sub_15E1220(v30, (__int64)&v132);
  v117[0] = v132;
  v117[1] = (unsigned int *)(unsigned int)v133;
  v31 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)(v31 + 16) )
    BUG();
  v32 = *(_QWORD *)(v31 + 24);
  v129 = (__int64 *)v131;
  v130 = 0x600000000LL;
  sub_15E2EC0(**(_QWORD **)(v32 + 16), v117, (__int64 *)&v129);
  v33 = *(_DWORD *)(v32 + 12);
  if ( v33 != 1 )
  {
    v106 = v4;
    v34 = 8;
    v101 = v5;
    v35 = v32;
    v36 = 8LL * (unsigned int)(v33 - 2) + 16;
    do
    {
      v37 = *(_QWORD *)(*(_QWORD *)(v35 + 16) + v34);
      v34 += 8;
      sub_15E2EC0(v37, v117, (__int64 *)&v129);
    }
    while ( v36 != v34 );
    v4 = v106;
    v5 = v101;
  }
  v38 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 56LL) + 40LL);
  v107 = **(__int64 ***)(*(_QWORD *)v4 + 16LL);
  if ( (_DWORD)v111 != 1 )
    v107 = sub_16463B0(v107, v111);
  *v129 = (__int64)v107;
  v102 = sub_15E26F0(v38, v103, v129, (unsigned int)v130);
  v40 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
  v41 = *(char *)(v4 + 23) < 0;
  v135 = (__int64 *)v137;
  v136 = 0x1000000000LL;
  if ( !v41 )
    goto LABEL_116;
  v42 = sub_1648A40(v4);
  v44 = v42 + v43;
  if ( *(char *)(v4 + 23) >= 0 )
  {
    if ( (unsigned int)(v44 >> 4) )
LABEL_156:
      BUG();
LABEL_116:
    v48 = 0;
    goto LABEL_58;
  }
  if ( !(unsigned int)((v44 - sub_1648A40(v4)) >> 4) )
    goto LABEL_116;
  if ( *(char *)(v4 + 23) >= 0 )
    goto LABEL_156;
  v45 = *(_DWORD *)(sub_1648A40(v4) + 8);
  if ( *(char *)(v4 + 23) >= 0 )
    BUG();
  v46 = sub_1648A40(v4);
  v48 = *(_DWORD *)(v46 + v47 - 4) - v45;
LABEL_58:
  v49 = v40 - 1 - v48;
  if ( v49 )
  {
    v50 = (unsigned int)v136;
    v51 = 0;
    do
    {
      v52 = *(_QWORD *)(v4 + 24 * (v51 - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
      if ( HIDWORD(v136) <= (unsigned int)v50 )
      {
        v100 = *(_QWORD *)(v4 + 24 * (v51 - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
        sub_16CD150((__int64)&v135, v137, 0, 8, v52, v39);
        v50 = (unsigned int)v136;
        v52 = v100;
      }
      ++v51;
      v135[v50] = v52;
      v50 = (unsigned int)(v136 + 1);
      LODWORD(v136) = v136 + 1;
    }
    while ( v49 != v51 );
  }
  if ( v108 )
    v135[a4] = v108;
  v53 = *(__int64 **)(a1 + 8);
  v54 = v53[1];
  v122 = v53;
  v123 = v54;
  v124 = v53[2];
  v55 = *v53;
  v125 = (unsigned __int8 *)v55;
  if ( v55 )
  {
    sub_1623A60((__int64)&v125, v55, 2);
    v53 = *(__int64 **)(a1 + 8);
  }
  v53[1] = *(_QWORD *)(v4 + 40);
  v53[2] = v4 + 24;
  v56 = *(_QWORD *)(v4 + 48);
  v126 = (unsigned __int8 *)v56;
  if ( v56 )
  {
    sub_1623A60((__int64)&v126, v56, 2);
    v57 = *v53;
    if ( !*v53 )
      goto LABEL_70;
    goto LABEL_69;
  }
  v57 = *v53;
  if ( *v53 )
  {
LABEL_69:
    sub_161E7C0((__int64)v53, v57);
LABEL_70:
    v58 = v126;
    *v53 = (__int64)v126;
    if ( v58 )
    {
      sub_1623210((__int64)&v126, v58, (__int64)v53);
    }
    else if ( v126 )
    {
      sub_161E7C0((__int64)&v126, (__int64)v126);
    }
  }
  v128[0] = 257;
  v59 = sub_172C570(*(_QWORD *)(a1 + 8), *(_QWORD *)(v102 + 24), v102, v135, (unsigned int)v136, (__int64 *)&v126, 0);
  sub_164B7C0(v59, v4);
  sub_15F4370(v59, v4, 0, 0);
  v60 = v99;
  if ( (_DWORD)v111 == 1 )
  {
    v128[0] = 257;
    v92 = *(_QWORD *)(a1 + 8);
    v93 = v5[2];
    if ( v93 <= 0x40 )
    {
      _RDX = *(_QWORD *)v5;
      __asm { tzcnt   rcx, rdx }
      v98 = 64;
      if ( *(_QWORD *)v5 )
        v98 = _RCX;
      if ( v98 <= v93 )
        v93 = v98;
      v94 = v93;
    }
    else
    {
      v94 = sub_16A58A0((__int64)v5);
    }
    v95 = sub_1599EF0(*(__int64 ***)v4);
    v24 = sub_17A3CD0(v92, v95, v59, v94, (__int64 *)&v126);
    goto LABEL_93;
  }
  v126 = (unsigned __int8 *)v128;
  v127 = 0x800000000LL;
  if ( !(_DWORD)v110 )
  {
    v68 = (char *)v128;
    v69 = 0;
    goto LABEL_86;
  }
  v112 = v59;
  v61 = 0;
  v62 = 0;
  v63 = v104;
  v64 = 8;
  v65 = 0;
  while ( 1 )
  {
    v67 = *(_QWORD *)v5;
    if ( v5[2] > 0x40 )
      v67 = *(_QWORD *)(v67 + 8LL * (v65 >> 6));
    if ( (v67 & (1LL << v65)) != 0 )
      break;
    if ( (unsigned int)v61 >= v64 )
    {
      v109 = v63;
      sub_16CD150((__int64)&v126, v128, 0, 4, v63, v60);
      v61 = (unsigned int)v127;
      v63 = v109;
    }
    ++v65;
    *(_DWORD *)&v126[4 * v61] = v63;
    v61 = (unsigned int)(v127 + 1);
    LODWORD(v127) = v127 + 1;
    if ( (_DWORD)v110 == v65 )
      goto LABEL_85;
LABEL_78:
    v64 = HIDWORD(v127);
  }
  v66 = v62 + 1;
  if ( (unsigned int)v61 >= v64 )
  {
    v105 = v63;
    sub_16CD150((__int64)&v126, v128, 0, 4, v63, v60);
    v61 = (unsigned int)v127;
    v63 = v105;
    v66 = v62 + 1;
  }
  ++v65;
  *(_DWORD *)&v126[4 * v61] = v62;
  v61 = (unsigned int)(v127 + 1);
  v62 = v66;
  LODWORD(v127) = v127 + 1;
  if ( (_DWORD)v110 != v65 )
    goto LABEL_78;
LABEL_85:
  v59 = v112;
  v68 = (char *)v126;
  v69 = (unsigned int)v61;
LABEL_86:
  v119 = 257;
  v70 = *(_QWORD *)(a1 + 8);
  v71 = sub_1599EF0((__int64 **)v107);
  v72 = (_BYTE *)sub_1599580(*(_QWORD *)(v70 + 24), v68, v69);
  if ( *(_BYTE *)(v59 + 16) > 0x10u || *(_BYTE *)(v71 + 16) > 0x10u || v72[16] > 0x10u )
  {
    v114 = v72;
    v121 = 257;
    v83 = (unsigned __int8 *)sub_1648A60(56, 3u);
    v24 = v83;
    if ( v83 )
      sub_15FA660((__int64)v83, (_QWORD *)v59, v71, v114, (__int64)v120, 0);
    v84 = *(_QWORD *)(v70 + 8);
    if ( v84 )
    {
      v85 = *(unsigned __int64 **)(v70 + 16);
      sub_157E9D0(v84 + 40, (__int64)v24);
      v86 = *((_QWORD *)v24 + 3);
      v87 = *v85;
      *((_QWORD *)v24 + 4) = v85;
      v87 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v24 + 3) = v87 | v86 & 7;
      *(_QWORD *)(v87 + 8) = v24 + 24;
      *v85 = *v85 & 7 | (unsigned __int64)(v24 + 24);
    }
    sub_164B780((__int64)v24, v118);
    v116 = v24;
    if ( !*(_QWORD *)(v70 + 80) )
      sub_4263D6(v24, v118, v88);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(v70 + 88))(v70 + 64, &v116);
    v89 = *(_QWORD *)v70;
    if ( *(_QWORD *)v70 )
    {
      v116 = *(unsigned __int8 **)v70;
      sub_1623A60((__int64)&v116, v89, 2);
      v90 = *((_QWORD *)v24 + 6);
      if ( v90 )
        sub_161E7C0((__int64)(v24 + 48), v90);
      v91 = v116;
      *((_QWORD *)v24 + 6) = v116;
      if ( v91 )
        sub_1623210((__int64)&v116, v91, (__int64)(v24 + 48));
    }
  }
  else
  {
    v24 = (unsigned __int8 *)sub_15A3950(v59, v71, v72, 0);
    v73 = sub_14DBA30((__int64)v24, *(_QWORD *)(v70 + 96), 0);
    if ( v73 )
      v24 = (unsigned __int8 *)v73;
  }
  if ( v126 != (unsigned __int8 *)v128 )
    _libc_free((unsigned __int64)v126);
LABEL_93:
  v74 = v123;
  v75 = v122;
  v76 = v124;
  if ( v123 )
  {
    v122[1] = v123;
    v75[2] = v76;
    if ( v76 != v74 + 40 )
    {
      if ( !v76 )
        BUG();
      v77 = *(_QWORD *)(v76 + 24);
      v126 = (unsigned __int8 *)v77;
      if ( v77 )
      {
        sub_1623A60((__int64)&v126, v77, 2);
        v78 = *v75;
        if ( *v75 )
          goto LABEL_98;
LABEL_99:
        v79 = v126;
        *v75 = (__int64)v126;
        if ( v79 )
        {
          sub_1623210((__int64)&v126, v79, (__int64)v75);
          v75 = v122;
        }
        else
        {
          if ( v126 )
            sub_161E7C0((__int64)&v126, (__int64)v126);
          v75 = v122;
        }
      }
      else
      {
        v78 = *v75;
        if ( *v75 )
        {
LABEL_98:
          sub_161E7C0((__int64)v75, v78);
          goto LABEL_99;
        }
      }
    }
  }
  else
  {
    v122[1] = 0;
    v75[2] = 0;
  }
  v126 = v125;
  if ( v125 )
  {
    sub_1623A60((__int64)&v126, (__int64)v125, 2);
    if ( v75 != (__int64 *)&v126 )
    {
      v80 = *v75;
      if ( *v75 )
        goto LABEL_104;
LABEL_105:
      v81 = v126;
      *v75 = (__int64)v126;
      if ( v81 )
      {
        sub_1623210((__int64)&v126, v81, (__int64)v75);
        v82 = (__int64)v125;
        goto LABEL_107;
      }
    }
LABEL_119:
    if ( v126 )
      sub_161E7C0((__int64)&v126, (__int64)v126);
    v82 = (__int64)v125;
LABEL_107:
    if ( v82 )
      sub_161E7C0((__int64)&v125, v82);
  }
  else if ( v75 != (__int64 *)&v126 )
  {
    v80 = *v75;
    if ( *v75 )
    {
LABEL_104:
      sub_161E7C0((__int64)v75, v80);
      goto LABEL_105;
    }
    goto LABEL_119;
  }
  if ( v135 != (__int64 *)v137 )
    _libc_free((unsigned __int64)v135);
  if ( v129 != (__int64 *)v131 )
    _libc_free((unsigned __int64)v129);
  if ( v132 != (unsigned int *)v134 )
    _libc_free((unsigned __int64)v132);
  return (__int64)v24;
}
