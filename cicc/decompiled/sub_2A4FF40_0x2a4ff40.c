// Function: sub_2A4FF40
// Address: 0x2a4ff40
//
__int64 __fastcall sub_2A4FF40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // r13
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 *v16; // rdx
  __int64 v17; // rcx
  _QWORD *v18; // r14
  unsigned __int8 *v19; // r15
  unsigned __int8 v20; // al
  char v21; // si
  __int64 *v22; // r14
  __int64 *v23; // r13
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r9
  __int64 *v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  char v32; // dl
  char v33; // al
  __int64 v34; // rdi
  unsigned int v35; // ecx
  unsigned __int8 **v36; // rdx
  unsigned __int8 *v37; // rax
  __int64 v38; // r14
  __int64 *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  _BYTE *v42; // rdi
  __int64 v43; // rax
  __int64 *v44; // rsi
  __int64 *v45; // rbx
  unsigned int v46; // r14d
  __int64 *v47; // r13
  __int64 v48; // r12
  unsigned __int8 **v49; // r10
  int v50; // r11d
  int v51; // eax
  int v52; // edx
  int v53; // eax
  int v54; // r11d
  unsigned int v55; // eax
  unsigned __int8 *v56; // rdi
  unsigned __int8 **v57; // rcx
  int v58; // eax
  unsigned __int8 **v59; // rdi
  unsigned int v60; // r13d
  unsigned __int8 *v61; // rcx
  int v62; // eax
  __int64 v63; // rsi
  __int64 v64; // r13
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdi
  __int64 v70; // r14
  __int64 *v71; // rax
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rax
  unsigned __int8 *v76; // r12
  _QWORD *v77; // r13
  unsigned __int64 v78; // r15
  __int64 v79; // rbx
  __int64 i; // r14
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // rdx
  int v84; // eax
  __int64 v85; // [rsp+0h] [rbp-4C0h]
  __int64 *v86; // [rsp+8h] [rbp-4B8h]
  int v87; // [rsp+8h] [rbp-4B8h]
  unsigned __int8 v88; // [rsp+8h] [rbp-4B8h]
  __int64 v90; // [rsp+28h] [rbp-498h]
  __int64 v91; // [rsp+28h] [rbp-498h]
  unsigned __int64 v92; // [rsp+28h] [rbp-498h]
  __int64 v93; // [rsp+28h] [rbp-498h]
  __int64 *v94; // [rsp+30h] [rbp-490h] BYREF
  __int64 v95; // [rsp+38h] [rbp-488h]
  _BYTE v96[256]; // [rsp+40h] [rbp-480h] BYREF
  __int64 *v97; // [rsp+140h] [rbp-380h] BYREF
  __int64 v98; // [rsp+148h] [rbp-378h]
  _QWORD v99[32]; // [rsp+150h] [rbp-370h] BYREF
  __int64 v100; // [rsp+250h] [rbp-270h] BYREF
  __int64 *v101; // [rsp+258h] [rbp-268h]
  __int64 v102; // [rsp+260h] [rbp-260h]
  int v103; // [rsp+268h] [rbp-258h]
  char v104; // [rsp+26Ch] [rbp-254h]
  char v105; // [rsp+270h] [rbp-250h] BYREF
  __int64 v106; // [rsp+370h] [rbp-150h] BYREF
  __int64 *v107; // [rsp+378h] [rbp-148h]
  __int64 v108; // [rsp+380h] [rbp-140h]
  int v109; // [rsp+388h] [rbp-138h]
  char v110; // [rsp+38Ch] [rbp-134h]
  char v111; // [rsp+390h] [rbp-130h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 16);
  v94 = (__int64 *)v96;
  v95 = 0x2000000000LL;
  v100 = 0;
  v101 = (__int64 *)&v105;
  v102 = 32;
  v103 = 0;
  v104 = 1;
  if ( !v7 )
    goto LABEL_33;
LABEL_2:
  v9 = v101;
  v10 = HIDWORD(v102);
  v11 = &v101[HIDWORD(v102)];
  if ( v101 == v11 )
  {
LABEL_27:
    if ( HIDWORD(v102) >= (unsigned int)v102 )
      goto LABEL_8;
    ++HIDWORD(v102);
    *v11 = v7;
    ++v100;
LABEL_9:
    v12 = (unsigned int)v95;
    v10 = HIDWORD(v95);
    v13 = (unsigned int)v95 + 1LL;
    if ( v13 > HIDWORD(v95) )
    {
      sub_C8D5F0((__int64)&v94, v96, v13, 8u, a5, a6);
      v12 = (unsigned int)v95;
    }
    v11 = v94;
    v94[v12] = v7;
    LODWORD(v95) = v95 + 1;
    v7 = *(_QWORD *)(v7 + 8);
    if ( v7 )
      goto LABEL_7;
  }
  else
  {
    while ( *v9 != v7 )
    {
      if ( v11 == ++v9 )
        goto LABEL_27;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        break;
LABEL_7:
      if ( v104 )
        goto LABEL_2;
LABEL_8:
      sub_C8CC70((__int64)&v100, v7, (__int64)v11, v10, a5, a6);
      if ( (_BYTE)v11 )
        goto LABEL_9;
    }
  }
  v14 = v95;
  if ( !(_DWORD)v95 )
    goto LABEL_32;
  v15 = (__int64)&v97;
  v6 = 0;
  while ( 1 )
  {
    v16 = v94;
    v17 = v14;
    v18 = (_QWORD *)v94[v14 - 1];
    LODWORD(v95) = v14 - 1;
    v19 = (unsigned __int8 *)v18[3];
    v20 = *v19;
    if ( *v19 <= 0x1Cu )
      goto LABEL_32;
    if ( v20 == 62 )
      break;
    if ( v20 == 85 )
    {
      v29 = *((_QWORD *)v19 - 4);
      if ( *(_BYTE *)v29 )
        goto LABEL_32;
      v30 = *(_QWORD *)(v29 + 24);
      if ( v30 == *((_QWORD *)v19 + 10) && (*(_BYTE *)(v29 + 33) & 0x20) != 0 )
      {
        if ( sub_B46A10(v18[3]) )
          goto LABEL_44;
        goto LABEL_32;
      }
      if ( v30 != *((_QWORD *)v19 + 10) )
        goto LABEL_32;
      if ( (*(_BYTE *)(v29 + 2) & 1) == 0 )
      {
        v63 = *(_QWORD *)(v29 + 96);
        v92 = v63 + 40LL * *(_QWORD *)(v29 + 104);
        goto LABEL_106;
      }
      sub_B2C6D0(*((_QWORD *)v19 - 4), v15, (__int64)v94, v17);
      v63 = *(_QWORD *)(v29 + 96);
      if ( (*(_BYTE *)(v29 + 2) & 1) != 0 )
      {
        v93 = *(_QWORD *)(v29 + 96);
        sub_B2C6D0(v29, v63, v81, v82);
        v83 = *(_QWORD *)(v29 + 96);
        v84 = *v19;
        v63 = v93;
      }
      else
      {
        v84 = *v19;
        v83 = *(_QWORD *)(v29 + 96);
      }
      v92 = v83 + 40LL * *(_QWORD *)(v29 + 104);
      switch ( v84 )
      {
        case '(':
          v64 = 32LL * (unsigned int)sub_B491D0((__int64)v19);
          break;
        case 'U':
LABEL_106:
          v64 = 0;
          break;
        case '"':
          v64 = 64;
          break;
        default:
          BUG();
      }
      if ( (v19[7] & 0x80u) != 0 )
      {
        v65 = sub_BD2BC0((__int64)v19);
        if ( (v19[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)((v65 + v66) >> 4) )
            goto LABEL_171;
        }
        else if ( (unsigned int)((v65 + v66 - sub_BD2BC0((__int64)v19)) >> 4) )
        {
          if ( (v19[7] & 0x80u) != 0 )
          {
            v87 = *(_DWORD *)(sub_BD2BC0((__int64)v19) + 8);
            if ( (v19[7] & 0x80u) == 0 )
              BUG();
            v67 = sub_BD2BC0((__int64)v19);
            v69 = 32LL * (unsigned int)(*(_DWORD *)(v67 + v68 - 4) - v87);
LABEL_137:
            v74 = *((_DWORD *)v19 + 1) & 0x7FFFFFF;
            v75 = (32 * v74 - 32 - v64 - v69) >> 5;
            if ( !(_DWORD)v75 )
              goto LABEL_44;
            v88 = v6;
            v76 = v19;
            v85 = a2;
            v77 = v18;
            v78 = v63;
            v79 = (unsigned int)(v75 - 1);
            for ( i = 0; *v77 != *(_QWORD *)&v76[32 * (i - v74)] || v92 > v78 && (unsigned __int8)sub_B2D680(v78); ++i )
            {
              v78 += 40LL;
              if ( v79 == i )
              {
                v19 = v76;
                a2 = v85;
                v6 = v88;
                goto LABEL_44;
              }
              v74 = *((_DWORD *)v76 + 1) & 0x7FFFFFF;
            }
            goto LABEL_32;
          }
LABEL_171:
          BUG();
        }
      }
      v69 = 0;
      goto LABEL_137;
    }
    if ( v20 != 78 )
    {
      if ( v20 == 63 )
      {
        if ( *v18 != *(_QWORD *)&v19[-32 * (*((_DWORD *)v19 + 1) & 0x7FFFFFF)] )
          goto LABEL_32;
        v38 = *((_QWORD *)v19 + 2);
        if ( !v38 )
          goto LABEL_44;
        if ( v104 )
        {
LABEL_55:
          v39 = v101;
          v17 = HIDWORD(v102);
          v16 = &v101[HIDWORD(v102)];
          if ( v101 == v16 )
          {
LABEL_126:
            if ( HIDWORD(v102) < (unsigned int)v102 )
            {
              ++HIDWORD(v102);
              *v16 = v38;
              ++v100;
              goto LABEL_62;
            }
            goto LABEL_61;
          }
          while ( *v39 != v38 )
          {
            if ( v16 == ++v39 )
              goto LABEL_126;
          }
        }
        else
        {
LABEL_61:
          while ( 1 )
          {
            sub_C8CC70((__int64)&v100, v38, (__int64)v16, v17, a5, a6);
            if ( !(_BYTE)v16 )
              break;
LABEL_62:
            v40 = (unsigned int)v95;
            v17 = HIDWORD(v95);
            v41 = (unsigned int)v95 + 1LL;
            if ( v41 > HIDWORD(v95) )
            {
              sub_C8D5F0((__int64)&v94, v96, v41, 8u, a5, a6);
              v40 = (unsigned int)v95;
            }
            v16 = v94;
            v94[v40] = v38;
            LODWORD(v95) = v95 + 1;
            v38 = *(_QWORD *)(v38 + 8);
            if ( !v38 )
              goto LABEL_44;
LABEL_60:
            if ( v104 )
              goto LABEL_55;
          }
        }
        v38 = *(_QWORD *)(v38 + 8);
        if ( !v38 )
          goto LABEL_44;
        goto LABEL_60;
      }
      if ( v20 == 61 )
        goto LABEL_44;
LABEL_32:
      v6 = 0;
      if ( !v104 )
        goto LABEL_48;
      goto LABEL_33;
    }
    v70 = *((_QWORD *)v19 + 2);
    if ( !v70 )
      goto LABEL_44;
    if ( v104 )
    {
LABEL_115:
      v71 = v101;
      v17 = HIDWORD(v102);
      v16 = &v101[HIDWORD(v102)];
      if ( v101 == v16 )
      {
LABEL_128:
        if ( HIDWORD(v102) < (unsigned int)v102 )
        {
          ++HIDWORD(v102);
          *v16 = v70;
          ++v100;
          goto LABEL_122;
        }
        goto LABEL_121;
      }
      while ( *v71 != v70 )
      {
        if ( v16 == ++v71 )
          goto LABEL_128;
      }
LABEL_119:
      v70 = *(_QWORD *)(v70 + 8);
      if ( !v70 )
        goto LABEL_44;
      goto LABEL_120;
    }
    while ( 1 )
    {
LABEL_121:
      sub_C8CC70((__int64)&v100, v70, (__int64)v16, v17, a5, a6);
      if ( !(_BYTE)v16 )
        goto LABEL_119;
LABEL_122:
      v72 = (unsigned int)v95;
      v17 = HIDWORD(v95);
      v73 = (unsigned int)v95 + 1LL;
      if ( v73 > HIDWORD(v95) )
      {
        sub_C8D5F0((__int64)&v94, v96, v73, 8u, a5, a6);
        v72 = (unsigned int)v95;
      }
      v16 = v94;
      v94[v72] = v70;
      LODWORD(v95) = v95 + 1;
      v70 = *(_QWORD *)(v70 + 8);
      if ( !v70 )
        break;
LABEL_120:
      if ( v104 )
        goto LABEL_115;
    }
LABEL_44:
    v15 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v15 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_91;
    }
    a6 = (unsigned int)(v15 - 1);
    v34 = *(_QWORD *)(a2 + 8);
    v35 = a6 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v36 = (unsigned __int8 **)(v34 + 8LL * v35);
    v37 = *v36;
    if ( v19 != *v36 )
    {
      v49 = 0;
      v50 = 1;
      while ( v37 != (unsigned __int8 *)-4096LL )
      {
        if ( v49 || v37 != (unsigned __int8 *)-8192LL )
          v36 = v49;
        v35 = a6 & (v50 + v35);
        a5 = v34 + 8LL * v35;
        v37 = *(unsigned __int8 **)a5;
        if ( v19 == *(unsigned __int8 **)a5 )
          goto LABEL_46;
        ++v50;
        v49 = v36;
        v36 = (unsigned __int8 **)(v34 + 8LL * v35);
      }
      v51 = *(_DWORD *)(a2 + 16);
      if ( !v49 )
        v49 = v36;
      ++*(_QWORD *)a2;
      v52 = v51 + 1;
      if ( 4 * (v51 + 1) >= (unsigned int)(3 * v15) )
      {
LABEL_91:
        sub_22EE7D0(a2, 2 * v15);
        v53 = *(_DWORD *)(a2 + 24);
        if ( !v53 )
          goto LABEL_172;
        v54 = v53 - 1;
        a6 = *(_QWORD *)(a2 + 8);
        v15 = *(unsigned int *)(a2 + 16);
        v55 = (v53 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v49 = (unsigned __int8 **)(a6 + 8LL * v55);
        v52 = v15 + 1;
        v56 = *v49;
        if ( v19 != *v49 )
        {
          v57 = 0;
          v15 = 1;
          while ( v56 != (unsigned __int8 *)-4096LL )
          {
            if ( !v57 && v56 == (unsigned __int8 *)-8192LL )
              v57 = v49;
            a5 = (unsigned int)(v15 + 1);
            v55 = v54 & (v15 + v55);
            v15 = v55;
            v49 = (unsigned __int8 **)(a6 + 8LL * v55);
            v56 = *v49;
            if ( v19 == *v49 )
              goto LABEL_86;
            v15 = (unsigned int)a5;
          }
          if ( v57 )
            v49 = v57;
        }
      }
      else if ( (int)v15 - *(_DWORD *)(a2 + 20) - v52 <= (unsigned int)v15 >> 3 )
      {
        sub_22EE7D0(a2, v15);
        v58 = *(_DWORD *)(a2 + 24);
        if ( !v58 )
        {
LABEL_172:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v15 = (unsigned int)(v58 - 1);
        a6 = *(_QWORD *)(a2 + 8);
        v59 = 0;
        v60 = v15 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v49 = (unsigned __int8 **)(a6 + 8LL * v60);
        v61 = *v49;
        v52 = *(_DWORD *)(a2 + 16) + 1;
        v62 = 1;
        if ( v19 != *v49 )
        {
          while ( v61 != (unsigned __int8 *)-4096LL )
          {
            if ( !v59 && v61 == (unsigned __int8 *)-8192LL )
              v59 = v49;
            a5 = (unsigned int)(v62 + 1);
            v60 = v15 & (v62 + v60);
            v49 = (unsigned __int8 **)(a6 + 8LL * v60);
            v61 = *v49;
            if ( v19 == *v49 )
              goto LABEL_86;
            ++v62;
          }
          if ( v59 )
            v49 = v59;
        }
      }
LABEL_86:
      *(_DWORD *)(a2 + 16) = v52;
      if ( *v49 != (unsigned __int8 *)-4096LL )
        --*(_DWORD *)(a2 + 20);
      *v49 = v19;
    }
LABEL_46:
    v14 = v95;
    if ( !(_DWORD)v95 )
      goto LABEL_47;
  }
  LOBYTE(v6) = *((_QWORD *)v19 - 4) == 0 || a1 != *((_QWORD *)v19 - 4);
  if ( (_BYTE)v6 )
    goto LABEL_32;
  v21 = 1;
  v22 = v99;
  v106 = 0;
  v98 = 0x2000000000LL;
  a5 = (__int64)&v106;
  v97 = v99;
  v23 = &v106;
  v107 = (__int64 *)&v111;
  v108 = 32;
  v109 = 0;
  v110 = 1;
  v24 = *((_QWORD *)v19 - 8);
  LODWORD(v98) = 1;
  v99[0] = v24;
  LODWORD(v24) = 1;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !(_DWORD)v24 )
      {
LABEL_25:
        v6 = 1;
        if ( v21 )
          goto LABEL_41;
        goto LABEL_26;
      }
      while ( 2 )
      {
        v25 = v97;
        v26 = (unsigned int)v24;
        v27 = v97[(unsigned int)v24 - 1];
        LODWORD(v98) = v24 - 1;
        if ( !v21 )
          goto LABEL_36;
        v28 = v107;
        v25 = &v107[HIDWORD(v108)];
        if ( v107 != v25 )
        {
          while ( v27 != *v28 )
          {
            if ( v25 == ++v28 )
              goto LABEL_66;
          }
LABEL_24:
          LODWORD(v24) = v98;
          if ( !(_DWORD)v98 )
            goto LABEL_25;
          continue;
        }
        break;
      }
LABEL_66:
      if ( HIDWORD(v108) >= (unsigned int)v108 )
      {
LABEL_36:
        v90 = v27;
        sub_C8CC70((__int64)v23, v27, (__int64)v25, v26, a5, v27);
        v21 = v110;
        v27 = v90;
        if ( v32 )
          goto LABEL_37;
        goto LABEL_24;
      }
      ++HIDWORD(v108);
      *v25 = v27;
      v21 = v110;
      ++v106;
LABEL_37:
      v33 = *(_BYTE *)v27;
      if ( *(_BYTE *)v27 <= 0x1Cu )
        goto LABEL_40;
      if ( v33 != 61 )
      {
        if ( v33 != 84 )
          goto LABEL_40;
        v43 = 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v27 + 7) & 0x40) != 0 )
        {
          a5 = *(_QWORD *)(v27 - 8);
          v27 = a5 + v43;
        }
        else
        {
          a5 = v27 - v43;
        }
        v24 = (unsigned int)v98;
        if ( a5 != v27 )
        {
          v91 = a2;
          v44 = v22;
          v45 = (__int64 *)a5;
          v46 = v6;
          v86 = v23;
          v47 = (__int64 *)v27;
          do
          {
            v48 = *v45;
            if ( v24 + 1 > (unsigned __int64)HIDWORD(v98) )
            {
              sub_C8D5F0((__int64)&v97, v44, v24 + 1, 8u, a5, v27);
              v24 = (unsigned int)v98;
            }
            v45 += 4;
            v97[v24] = v48;
            v24 = (unsigned int)(v98 + 1);
            LODWORD(v98) = v98 + 1;
          }
          while ( v47 != v45 );
          v6 = v46;
          a2 = v91;
          v22 = v44;
          v23 = v86;
          v21 = v110;
        }
        continue;
      }
      break;
    }
    v42 = *(_BYTE **)(v27 - 32);
    if ( *v42 != 22 )
      goto LABEL_40;
    if ( (unsigned __int8)sub_B2D680((__int64)v42) )
    {
      LODWORD(v24) = v98;
      v21 = v110;
      continue;
    }
    break;
  }
  v21 = v110;
  v6 = 0;
LABEL_40:
  if ( !v21 )
LABEL_26:
    _libc_free((unsigned __int64)v107);
LABEL_41:
  if ( v97 != v22 )
    _libc_free((unsigned __int64)v97);
  if ( (_BYTE)v6 )
    goto LABEL_44;
LABEL_47:
  if ( !v104 )
LABEL_48:
    _libc_free((unsigned __int64)v101);
LABEL_33:
  if ( v94 != (__int64 *)v96 )
    _libc_free((unsigned __int64)v94);
  return v6;
}
