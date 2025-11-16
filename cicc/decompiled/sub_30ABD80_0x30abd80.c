// Function: sub_30ABD80
// Address: 0x30abd80
//
__int64 __fastcall sub_30ABD80(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4, char a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // r9
  unsigned __int8 *v10; // r14
  unsigned __int8 **v11; // rax
  unsigned __int8 **v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // ecx
  __int64 v16; // r12
  int v17; // r13d
  bool v18; // of
  unsigned __int64 v19; // r12
  unsigned int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 result; // rax
  int v26; // r15d
  __int64 v27; // rdx
  __int64 v28; // r12
  int v29; // eax
  unsigned __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r8
  unsigned __int8 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r10
  __int64 v37; // r15
  unsigned __int8 **v38; // rcx
  int v39; // esi
  unsigned __int8 **v40; // rdx
  unsigned __int64 v41; // rax
  int v42; // edx
  __int64 v43; // r9
  __int64 v44; // r15
  unsigned __int64 v45; // rax
  __int64 v46; // r8
  unsigned __int8 *v47; // rax
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // r15
  unsigned __int8 **v51; // rcx
  int v52; // esi
  unsigned __int8 **v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // r15
  int v56; // edx
  int v57; // r14d
  unsigned __int64 v58; // rax
  bool v59; // cc
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r15
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // r14
  __int64 v66; // r15
  __int64 v67; // rax
  unsigned int *v68; // rax
  char v69; // r15
  __int64 v70; // rax
  __int64 v71; // r8
  unsigned __int8 *v72; // rax
  __int64 v73; // rax
  __int64 v74; // r10
  __int64 v75; // r15
  unsigned __int8 **v76; // rcx
  int v77; // esi
  unsigned __int8 **v78; // rdx
  unsigned __int64 v79; // rax
  int v80; // edx
  __int64 v81; // r15
  unsigned __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rbx
  __int64 v86; // rsi
  _QWORD *v87; // rax
  _QWORD *v88; // rdx
  int v89; // r11d
  __int64 v90; // r10
  int v91; // edi
  int v92; // ecx
  int v93; // r8d
  int v94; // r8d
  __int64 v95; // r9
  unsigned int v96; // edx
  _QWORD *v97; // rdi
  int v98; // esi
  __int64 v99; // r10
  int v100; // r8d
  int v101; // r8d
  __int64 v102; // r9
  int v103; // edx
  unsigned int v104; // r14d
  __int64 v105; // rdi
  _QWORD *v106; // rsi
  __int64 v107; // [rsp+0h] [rbp-C0h]
  int v110; // [rsp+14h] [rbp-ACh]
  __int64 v111; // [rsp+18h] [rbp-A8h]
  __int64 v112; // [rsp+20h] [rbp-A0h]
  __int64 v113; // [rsp+20h] [rbp-A0h]
  __int64 v114; // [rsp+20h] [rbp-A0h]
  __int64 v115; // [rsp+28h] [rbp-98h]
  __int64 v116; // [rsp+28h] [rbp-98h]
  __int64 v117; // [rsp+28h] [rbp-98h]
  int v119; // [rsp+38h] [rbp-88h]
  __int64 v120; // [rsp+38h] [rbp-88h]
  __int64 v121; // [rsp+38h] [rbp-88h]
  __int64 v122; // [rsp+38h] [rbp-88h]
  int v123; // [rsp+38h] [rbp-88h]
  __int64 v124; // [rsp+38h] [rbp-88h]
  __int64 v125; // [rsp+38h] [rbp-88h]
  _QWORD *v126; // [rsp+48h] [rbp-78h]
  unsigned __int8 **v129; // [rsp+60h] [rbp-60h] BYREF
  __int64 v130; // [rsp+68h] [rbp-58h]
  _BYTE v131[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 16);
  ++*(_DWORD *)(a1 + 48);
  v8 = (_QWORD *)a2[7];
  v111 = v7;
  v110 = *(_DWORD *)(a1 + 24);
  v126 = a2 + 6;
  if ( v8 == a2 + 6 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v9 = (__int64)(v8 - 3);
      v10 = 0;
      if ( v8 )
        v10 = (unsigned __int8 *)(v8 - 3);
      if ( *(_BYTE *)(a4 + 28) )
        break;
      if ( !sub_C8CA60(a4, (__int64)v10) )
        goto LABEL_22;
LABEL_9:
      v8 = (_QWORD *)v8[1];
      if ( v126 == v8 )
        goto LABEL_10;
    }
    v11 = *(unsigned __int8 ***)(a4 + 8);
    v12 = &v11[*(unsigned int *)(a4 + 20)];
    if ( v11 != v12 )
    {
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_22;
      }
      goto LABEL_9;
    }
LABEL_22:
    v26 = *v10;
    if ( (unsigned __int8)(v26 - 34) > 0x33u )
      goto LABEL_71;
    v27 = 0x8000000000041LL;
    if ( !_bittest64(&v27, (unsigned int)(v26 - 34)) )
      goto LABEL_28;
    v28 = *((_QWORD *)v10 - 4);
    if ( !*(_BYTE *)v28 )
    {
      if ( *((_QWORD *)v10 + 10) != *(_QWORD *)(v28 + 24) )
        goto LABEL_26;
      v69 = sub_DF9C30(a3, *((_BYTE **)v10 - 4));
      if ( (unsigned __int8)sub_A73ED0((_QWORD *)v10 + 9, 31)
        || (unsigned __int8)sub_B49560((__int64)v10, 31) == 1
        || !v69 )
      {
        if ( a2[9] == v28 )
          goto LABEL_131;
        if ( !v69 )
          goto LABEL_27;
      }
      else
      {
        if ( (*(_BYTE *)(v28 + 32) & 0xF) == 7 && sub_AD0010(v28) || a5 )
          ++*(_DWORD *)(v6 + 124);
        if ( v28 == a2[9] )
        {
LABEL_131:
          *(_BYTE *)(v6 + 1) = 1;
          if ( !v69 )
            goto LABEL_27;
        }
      }
      ++*(_DWORD *)(v6 + 88);
      if ( !sub_B2FC80(v28) || (*(_BYTE *)(v28 + 33) & 0x20) != 0 )
        goto LABEL_27;
      ++*(_DWORD *)(v6 + 92);
      v26 = *v10;
      goto LABEL_28;
    }
    if ( *(_BYTE *)v28 != 25 )
    {
LABEL_26:
      ++*(_DWORD *)(v6 + 88);
      ++*(_DWORD *)(v6 + 96);
LABEL_27:
      v26 = *v10;
    }
LABEL_28:
    if ( (_BYTE)v26 == 62 )
    {
      ++*(_DWORD *)(v6 + 108);
      v26 = *v10;
      goto LABEL_30;
    }
LABEL_71:
    if ( (unsigned int)(unsigned __int8)v26 - 42 > 0x11 )
    {
      if ( sub_CEA640((__int64)v10) )
      {
        ++*(_DWORD *)(v6 + 112);
        v26 = *v10;
      }
      else
      {
        v26 = *v10;
        if ( (_BYTE)v26 == 61 )
        {
          v70 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
          if ( (v10[7] & 0x40) != 0 )
          {
            v71 = *((_QWORD *)v10 - 1);
            v72 = (unsigned __int8 *)(v71 + v70);
          }
          else
          {
            v71 = (__int64)&v10[-v70];
            v72 = v10;
          }
          v73 = (__int64)&v72[-v71];
          v74 = v73 >> 5;
          v129 = (unsigned __int8 **)v131;
          v130 = 0x400000000LL;
          v75 = v73 >> 5;
          if ( (unsigned __int64)v73 > 0x80 )
          {
            v114 = v73;
            v117 = v71;
            v125 = v73 >> 5;
            sub_C8D5F0((__int64)&v129, v131, v73 >> 5, 8u, v71, v9);
            v78 = v129;
            v77 = v130;
            LODWORD(v74) = v125;
            v71 = v117;
            v73 = v114;
            v76 = &v129[(unsigned int)v130];
          }
          else
          {
            v76 = (unsigned __int8 **)v131;
            v77 = 0;
            v78 = (unsigned __int8 **)v131;
          }
          if ( v73 > 0 )
          {
            v79 = 0;
            do
            {
              v76[v79 / 8] = *(unsigned __int8 **)(v71 + 4 * v79);
              v79 += 8LL;
              --v75;
            }
            while ( v75 );
            v78 = v129;
            v77 = v130;
          }
          LODWORD(v130) = v77 + v74;
          v81 = sub_DFCEF0((__int64 **)a3, v10, v78, (unsigned int)(v77 + v74), 1);
          if ( v129 != (unsigned __int8 **)v131 )
          {
            v123 = v80;
            _libc_free((unsigned __int64)v129);
            v80 = v123;
          }
          if ( v80 )
            v81 = v107;
          v107 = v81;
          if ( (unsigned int)v81 <= 7 )
            ++*(_DWORD *)(v6 + 104);
          else
            ++*(_DWORD *)(v6 + 100);
          v26 = *v10;
        }
      }
    }
    else
    {
      if ( (_BYTE)v26 == 42 )
      {
        ++*(_DWORD *)(v6 + 116);
        v26 = *v10;
      }
      if ( (_BYTE)v26 == 46 )
      {
        ++*(_DWORD *)(v6 + 120);
        v26 = *v10;
      }
    }
LABEL_30:
    if ( (_BYTE)v26 == 60 )
    {
      if ( !sub_B4D040((__int64)v10) )
        *(_BYTE *)(v6 + 8) = 1;
      v26 = *v10;
    }
    if ( (_BYTE)v26 == 90 || (v29 = *(unsigned __int8 *)(*((_QWORD *)v10 + 1) + 8LL), (unsigned int)(v29 - 17) <= 1) )
    {
      ++*(_DWORD *)(v6 + 128);
      v26 = *v10;
      LOBYTE(v29) = *(_BYTE *)(*((_QWORD *)v10 + 1) + 8LL);
    }
    if ( (_BYTE)v29 != 11 )
      goto LABEL_38;
    if ( (_BYTE)v26 != 85 )
    {
      if ( (unsigned __int8)sub_B463C0((__int64)v10, (__int64)a2) )
      {
        *(_BYTE *)(v6 + 2) = 1;
        v26 = *v10;
      }
LABEL_38:
      v30 = (unsigned int)(v26 - 34);
      if ( (unsigned __int8)v30 > 0x33u )
        goto LABEL_43;
      v31 = 0x8000000000041LL;
      if ( !_bittest64(&v31, v30) )
        goto LABEL_43;
      goto LABEL_40;
    }
    v83 = *((_QWORD *)v10 - 4);
    if ( (!v83
       || *(_BYTE *)v83
       || *(_QWORD *)(v83 + 24) != *((_QWORD *)v10 + 10)
       || (*(_BYTE *)(v83 + 33) & 0x20) == 0
       || (unsigned int)(*(_DWORD *)(v83 + 36) - 142) > 2)
      && (unsigned __int8)sub_B463C0((__int64)v10, (__int64)a2) )
    {
      *(_BYTE *)(v6 + 2) = 1;
      v26 = *v10;
      goto LABEL_38;
    }
LABEL_40:
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)v10 + 9, 27) || (unsigned __int8)sub_B49560((__int64)v10, 27) )
    {
      v59 = *(_DWORD *)(v6 + 4) <= 1;
      *(_BYTE *)(v6 + 2) = 1;
      if ( !v59 )
        goto LABEL_43;
    }
    else if ( *(int *)(v6 + 4) > 1 )
    {
      goto LABEL_43;
    }
    if ( !(unsigned __int8)sub_A73ED0((_QWORD *)v10 + 9, 6) && !(unsigned __int8)sub_B49560((__int64)v10, 6) )
      goto LABEL_43;
    if ( *v10 == 85
      && (v84 = *((_QWORD *)v10 - 4)) != 0
      && !*(_BYTE *)v84
      && *(_QWORD *)(v84 + 24) == *((_QWORD *)v10 + 10)
      && (*(_BYTE *)(v84 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v84 + 36) - 142) <= 2 )
    {
      if ( !a6 )
        goto LABEL_93;
    }
    else
    {
      if ( (v10[7] & 0x80u) == 0 )
        goto LABEL_168;
      v60 = sub_BD2BC0((__int64)v10);
      v62 = v60 + v61;
      if ( (v10[7] & 0x80u) == 0 )
        v63 = v62 >> 4;
      else
        LODWORD(v63) = (v62 - sub_BD2BC0((__int64)v10)) >> 4;
      if ( !(_DWORD)v63 )
        goto LABEL_168;
      v120 = v6;
      v64 = (__int64)v10;
      v65 = 0;
      v66 = 16LL * (unsigned int)v63;
      while ( 1 )
      {
        v67 = 0;
        if ( *(char *)(v64 + 7) < 0 )
          v67 = sub_BD2BC0(v64);
        v68 = (unsigned int *)(v65 + v67);
        if ( *(_DWORD *)(*(_QWORD *)v68 + 8LL) == 9 )
          break;
        v65 += 16;
        if ( v65 == v66 )
        {
          v10 = (unsigned __int8 *)v64;
          v6 = v120;
          goto LABEL_168;
        }
      }
      v10 = (unsigned __int8 *)v64;
      v6 = v120;
      if ( !*(_QWORD *)&v10[32 * (v68[2] - (unsigned __int64)(*((_DWORD *)v10 + 1) & 0x7FFFFFF))] )
      {
LABEL_168:
        *(_DWORD *)(v6 + 4) = 3;
        goto LABEL_43;
      }
      if ( !a6 || *v10 != 85 || (v84 = *((_QWORD *)v10 - 4)) == 0 )
      {
LABEL_93:
        *(_DWORD *)(v6 + 4) = 1;
        goto LABEL_43;
      }
    }
    if ( *(_BYTE *)v84
      || *(_QWORD *)(v84 + 24) != *((_QWORD *)v10 + 10)
      || (*(_BYTE *)(v84 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v84 + 36) - 142) > 2
      || !*((_QWORD *)v10 + 2) )
    {
      goto LABEL_93;
    }
    v124 = v6;
    v85 = *((_QWORD *)v10 + 2);
    while ( 1 )
    {
      v86 = *(_QWORD *)(*(_QWORD *)(v85 + 24) + 40LL);
      if ( !*(_BYTE *)(a6 + 84) )
        break;
      v87 = *(_QWORD **)(a6 + 64);
      v88 = &v87[*(unsigned int *)(a6 + 76)];
      if ( v87 == v88 )
        goto LABEL_183;
      while ( v86 != *v87 )
      {
        if ( v88 == ++v87 )
          goto LABEL_183;
      }
LABEL_165:
      v85 = *(_QWORD *)(v85 + 8);
      if ( !v85 )
      {
        v6 = v124;
        goto LABEL_93;
      }
    }
    if ( sub_C8CA60(a6 + 56, v86) )
      goto LABEL_165;
LABEL_183:
    v6 = v124;
    *(_DWORD *)(v124 + 4) = 2;
LABEL_43:
    v32 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
    if ( (v10[7] & 0x40) != 0 )
    {
      v33 = *((_QWORD *)v10 - 1);
      v34 = (unsigned __int8 *)(v33 + v32);
    }
    else
    {
      v33 = (__int64)&v10[-v32];
      v34 = v10;
    }
    v35 = (__int64)&v34[-v33];
    v129 = (unsigned __int8 **)v131;
    v130 = 0x400000000LL;
    v36 = v35 >> 5;
    v37 = v35 >> 5;
    if ( (unsigned __int64)v35 > 0x80 )
    {
      v112 = v35;
      v115 = v33;
      v121 = v35 >> 5;
      sub_C8D5F0((__int64)&v129, v131, v35 >> 5, 8u, v33, v9);
      v40 = v129;
      v39 = v130;
      LODWORD(v36) = v121;
      v33 = v115;
      v35 = v112;
      v38 = &v129[(unsigned int)v130];
    }
    else
    {
      v38 = (unsigned __int8 **)v131;
      v39 = 0;
      v40 = (unsigned __int8 **)v131;
    }
    if ( v35 > 0 )
    {
      v41 = 0;
      do
      {
        v38[v41 / 8] = *(unsigned __int8 **)(v33 + 4 * v41);
        v41 += 8LL;
        --v37;
      }
      while ( v37 );
      v40 = v129;
      v39 = v130;
    }
    LODWORD(v130) = v39 + v36;
    v44 = sub_DFCEF0((__int64 **)a3, v10, v40, (unsigned int)(v39 + v36), 4);
    if ( v129 != (unsigned __int8 **)v131 )
    {
      v119 = v42;
      _libc_free((unsigned __int64)v129);
      v42 = v119;
    }
    if ( v42 == 1 )
      *(_DWORD *)(v6 + 40) = 1;
    v45 = *(_QWORD *)(v6 + 32) + v44;
    if ( __OFADD__(*(_QWORD *)(v6 + 32), v44) )
    {
      v45 = 0x8000000000000000LL;
      if ( v44 > 0 )
        v45 = 0x7FFFFFFFFFFFFFFFLL;
    }
    *(_QWORD *)(v6 + 32) = v45;
    if ( (v10[7] & 0x40) != 0 )
    {
      v46 = *((_QWORD *)v10 - 1);
      v47 = (unsigned __int8 *)(v46 + 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF));
    }
    else
    {
      v46 = (__int64)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
      v47 = v10;
    }
    v48 = (__int64)&v47[-v46];
    v129 = (unsigned __int8 **)v131;
    v130 = 0x400000000LL;
    v49 = v48 >> 5;
    v50 = v48 >> 5;
    if ( (unsigned __int64)v48 > 0x80 )
    {
      v113 = v48;
      v116 = v46;
      v122 = v48 >> 5;
      sub_C8D5F0((__int64)&v129, v131, v48 >> 5, 8u, v46, v43);
      v53 = v129;
      v52 = v130;
      LODWORD(v49) = v122;
      v46 = v116;
      v48 = v113;
      v51 = &v129[(unsigned int)v130];
    }
    else
    {
      v51 = (unsigned __int8 **)v131;
      v52 = 0;
      v53 = (unsigned __int8 **)v131;
    }
    if ( v48 > 0 )
    {
      v54 = 0;
      do
      {
        v51[v54 / 8] = *(unsigned __int8 **)(v46 + 4 * v54);
        v54 += 8LL;
        --v50;
      }
      while ( v50 );
      v53 = v129;
      v52 = v130;
    }
    LODWORD(v130) = v52 + v49;
    v55 = sub_DFCEF0((__int64 **)a3, v10, v53, (unsigned int)(v52 + v49), 2);
    v57 = v56;
    if ( v129 != (unsigned __int8 **)v131 )
      _libc_free((unsigned __int64)v129);
    if ( v57 == 1 )
      *(_DWORD *)(v6 + 24) = 1;
    v58 = *(_QWORD *)(v6 + 16) + v55;
    if ( __OFADD__(*(_QWORD *)(v6 + 16), v55) )
    {
      v58 = 0x8000000000000000LL;
      if ( v55 > 0 )
        v58 = 0x7FFFFFFFFFFFFFFFLL;
    }
    *(_QWORD *)(v6 + 16) = v58;
    v8 = (_QWORD *)v8[1];
  }
  while ( v126 != v8 );
LABEL_10:
  v13 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  v14 = v13;
  if ( v126 == (_QWORD *)v13 )
    goto LABEL_146;
  if ( !v13 )
    goto LABEL_225;
  v15 = *(unsigned __int8 *)(v13 - 24);
  if ( (unsigned int)(v15 - 30) > 0xA )
LABEL_146:
    BUG();
  if ( (_BYTE)v15 == 30
    && ((++*(_DWORD *)(v6 + 132), v82 = a2[6] & 0xFFFFFFFFFFFFFFF8LL, v14 = v82, v126 == (_QWORD *)v82) || !v82)
    || (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
  {
LABEL_225:
    BUG();
  }
  v16 = *(_QWORD *)(v6 + 16);
  v17 = 1;
  *(_BYTE *)(v6 + 2) |= *(_BYTE *)(v14 - 24) == 33;
  if ( v110 != 1 )
    v17 = *(_DWORD *)(v6 + 24);
  v18 = __OFSUB__(v16, v111);
  v19 = v16 - v111;
  if ( v18 )
  {
    v19 = 0x8000000000000000LL;
    if ( v111 <= 0 )
      v19 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v20 = *(_DWORD *)(v6 + 80);
  if ( !v20 )
  {
    ++*(_QWORD *)(v6 + 56);
    goto LABEL_185;
  }
  v21 = *(_QWORD *)(v6 + 64);
  LODWORD(v22) = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v23 = v21 + 24LL * (unsigned int)v22;
  v24 = *(_QWORD **)v23;
  if ( a2 == *(_QWORD **)v23 )
    goto LABEL_20;
  v89 = 1;
  v90 = 0;
  while ( v24 != (_QWORD *)-4096LL )
  {
    if ( v24 == (_QWORD *)-8192LL && !v90 )
      v90 = v23;
    v22 = (v20 - 1) & ((_DWORD)v22 + v89);
    v23 = v21 + 24 * v22;
    v24 = *(_QWORD **)v23;
    if ( *(_QWORD **)v23 == a2 )
      goto LABEL_20;
    ++v89;
  }
  v91 = *(_DWORD *)(v6 + 72);
  if ( v90 )
    v23 = v90;
  ++*(_QWORD *)(v6 + 56);
  v92 = v91 + 1;
  if ( 4 * (v91 + 1) >= 3 * v20 )
  {
LABEL_185:
    sub_30ABB80(v6 + 56, 2 * v20);
    v93 = *(_DWORD *)(v6 + 80);
    if ( v93 )
    {
      v94 = v93 - 1;
      v95 = *(_QWORD *)(v6 + 64);
      v92 = *(_DWORD *)(v6 + 72) + 1;
      v96 = v94 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = v95 + 24LL * v96;
      v97 = *(_QWORD **)v23;
      if ( *(_QWORD **)v23 != a2 )
      {
        v98 = 1;
        v99 = 0;
        while ( v97 != (_QWORD *)-4096LL )
        {
          if ( !v99 && v97 == (_QWORD *)-8192LL )
            v99 = v23;
          v96 = v94 & (v98 + v96);
          v23 = v95 + 24LL * v96;
          v97 = *(_QWORD **)v23;
          if ( *(_QWORD **)v23 == a2 )
            goto LABEL_177;
          ++v98;
        }
        if ( v99 )
          v23 = v99;
      }
      goto LABEL_177;
    }
    goto LABEL_226;
  }
  if ( v20 - *(_DWORD *)(v6 + 76) - v92 <= v20 >> 3 )
  {
    sub_30ABB80(v6 + 56, v20);
    v100 = *(_DWORD *)(v6 + 80);
    if ( v100 )
    {
      v101 = v100 - 1;
      v102 = *(_QWORD *)(v6 + 64);
      v103 = 1;
      v104 = v101 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v92 = *(_DWORD *)(v6 + 72) + 1;
      v105 = 0;
      v23 = v102 + 24LL * v104;
      v106 = *(_QWORD **)v23;
      if ( *(_QWORD **)v23 != a2 )
      {
        while ( v106 != (_QWORD *)-4096LL )
        {
          if ( v106 == (_QWORD *)-8192LL && !v105 )
            v105 = v23;
          v104 = v101 & (v103 + v104);
          v23 = v102 + 24LL * v104;
          v106 = *(_QWORD **)v23;
          if ( *(_QWORD **)v23 == a2 )
            goto LABEL_177;
          ++v103;
        }
        if ( v105 )
          v23 = v105;
      }
      goto LABEL_177;
    }
LABEL_226:
    ++*(_DWORD *)(v6 + 72);
    BUG();
  }
LABEL_177:
  *(_DWORD *)(v6 + 72) = v92;
  if ( *(_QWORD *)v23 != -4096 )
    --*(_DWORD *)(v6 + 76);
  *(_QWORD *)(v23 + 8) = 0;
  *(_DWORD *)(v23 + 16) = 0;
  *(_QWORD *)v23 = a2;
LABEL_20:
  *(_QWORD *)(v23 + 8) = v19;
  result = v23 + 8;
  *(_DWORD *)(result + 8) = v17;
  return result;
}
