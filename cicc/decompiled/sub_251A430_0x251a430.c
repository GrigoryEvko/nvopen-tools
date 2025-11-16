// Function: sub_251A430
// Address: 0x251a430
//
__int64 __fastcall sub_251A430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 j; // rbx
  int v7; // r12d
  __int64 v8; // rax
  _BYTE *v9; // rsi
  unsigned int v10; // esi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  int v14; // r10d
  unsigned int v15; // edi
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // r12
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  _QWORD *v26; // rax
  unsigned int v27; // esi
  __int64 v28; // r8
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 *v31; // rdx
  __int64 v32; // rcx
  __int64 *v33; // rdi
  _BYTE *v34; // rax
  __int64 i; // rdx
  int v36; // r10d
  __int64 *v37; // rdx
  unsigned int v38; // edi
  __int64 *v39; // rax
  __int64 v40; // rcx
  __int64 *v41; // r13
  __int16 v42; // ax
  __int16 v43; // ax
  __int64 v44; // r13
  unsigned int v45; // ecx
  int v46; // eax
  __int64 v47; // rdi
  bool v48; // zf
  __int16 v49; // ax
  unsigned int v50; // esi
  __int64 v51; // rax
  __int64 v52; // r9
  int v53; // r11d
  __int64 *v54; // r10
  __int64 v55; // r8
  unsigned int v56; // edx
  __int64 *v57; // rcx
  __int64 v58; // rdi
  __int64 v59; // rdx
  _QWORD *v60; // r13
  _QWORD *v61; // rbx
  _BYTE *v62; // r15
  __int64 *v63; // r8
  unsigned int v64; // r15d
  int v65; // r9d
  __int64 v66; // rsi
  __int64 *v67; // rbx
  int v68; // ecx
  int v69; // ecx
  __int64 v70; // rax
  __int64 v71; // r13
  int v72; // r10d
  int v73; // r10d
  __int64 v74; // r8
  __int64 v75; // rcx
  int v76; // esi
  int v77; // r11d
  __int64 v78; // rdi
  int v79; // r9d
  int v80; // r9d
  __int64 v81; // rdi
  __int64 v82; // rsi
  __int64 v83; // r15
  int v84; // r10d
  int v85; // ecx
  int v86; // r11d
  __int64 *v87; // r9
  int v88; // eax
  int v89; // edx
  __int64 v90; // rcx
  __int64 v91; // r12
  __int64 v92; // rax
  int v93; // r10d
  __int64 *v94; // r9
  __int64 v95; // [rsp+0h] [rbp-F0h]
  __int64 v96; // [rsp+10h] [rbp-E0h]
  __int64 v97; // [rsp+18h] [rbp-D8h]
  __int64 v101; // [rsp+48h] [rbp-A8h]
  __int64 v102; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v103; // [rsp+58h] [rbp-98h] BYREF
  __int64 v104; // [rsp+60h] [rbp-90h] BYREF
  __int64 v105; // [rsp+68h] [rbp-88h]
  __int64 v106; // [rsp+70h] [rbp-80h]
  unsigned int v107; // [rsp+78h] [rbp-78h]
  __int64 *v108; // [rsp+80h] [rbp-70h] BYREF
  __int64 v109; // [rsp+88h] [rbp-68h]
  _QWORD v110[12]; // [rsp+90h] [rbp-60h] BYREF

  v3 = a2 + 72;
  v106 = 0;
  *(_BYTE *)(a3 + 114) = sub_B2D620(a2, "kernel", 6u);
  v4 = *(_QWORD *)(a2 + 80);
  v104 = 0;
  v105 = 0;
  v107 = 0;
  if ( a2 + 72 == v4 )
  {
    j = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v4 + 32);
      if ( j != v4 + 24 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        goto LABEL_7;
      if ( !v4 )
        BUG();
    }
  }
  if ( v3 == v4 )
    goto LABEL_7;
  do
  {
    if ( !j )
      BUG();
    v101 = j - 24;
    v7 = *(unsigned __int8 *)(j - 24) - 29;
    switch ( *(_BYTE *)(j - 24) )
    {
      case 0x1E:
      case 0x1F:
      case 0x22:
      case 0x23:
      case 0x25:
      case 0x27:
      case 0x28:
      case 0x3C:
      case 0x3D:
      case 0x3E:
      case 0x41:
      case 0x42:
      case 0x4F:
        goto LABEL_20;
      case 0x55:
        v8 = *(_QWORD *)(j - 56);
        if ( v8
          && !*(_BYTE *)v8
          && *(_QWORD *)(v8 + 24) == *(_QWORD *)(j + 56)
          && (*(_BYTE *)(v8 + 33) & 0x20) != 0
          && *(_DWORD *)(v8 + 36) == 11 )
        {
          v103 = (__int64 *)(j - 24);
          v95 = a1 + 160;
          v27 = *(_DWORD *)(a1 + 184);
          if ( v27 )
          {
            v28 = v27 - 1;
            v29 = *(_QWORD *)(a1 + 168);
            v30 = v28 & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
            v31 = (__int64 *)(v29 + 8LL * v30);
            v32 = *v31;
            if ( v101 == *v31 )
            {
LABEL_60:
              sub_CFA400(v101, a1 + 128);
              v33 = v110;
              v34 = *(_BYTE **)(j - 32LL * (*(_DWORD *)(j - 20) & 0x7FFFFFF) - 24);
              v108 = v110;
              v109 = 0x600000000LL;
              if ( *v34 <= 0x1Cu )
                goto LABEL_19;
              v110[0] = v34;
              LODWORD(i) = 1;
              LODWORD(v109) = 1;
              v97 = v3;
              v96 = j;
              while ( 1 )
              {
                v44 = v33[(unsigned int)i - 1];
                LODWORD(v109) = i - 1;
                v102 = v44;
                if ( !v107 )
                  break;
                v36 = 1;
                v37 = 0;
                v38 = (v107 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                v39 = (__int64 *)(v105 + 16LL * v38);
                v40 = *v39;
                if ( *v39 != v44 )
                {
                  while ( v40 != -4096 )
                  {
                    if ( !v37 && v40 == -8192 )
                      v37 = v39;
                    v38 = (v107 - 1) & (v36 + v38);
                    v39 = (__int64 *)(v105 + 16LL * v38);
                    v40 = *v39;
                    if ( v44 == *v39 )
                      goto LABEL_63;
                    ++v36;
                  }
                  if ( !v37 )
                    v37 = v39;
                  ++v104;
                  v46 = v106 + 1;
                  if ( 4 * ((int)v106 + 1) < 3 * v107 )
                  {
                    if ( v107 - HIDWORD(v106) - v46 <= v107 >> 3 )
                    {
                      sub_2513E70((__int64)&v104, v107);
                      if ( !v107 )
                      {
LABEL_189:
                        LODWORD(v106) = v106 + 1;
                        BUG();
                      }
                      v63 = 0;
                      v64 = (v107 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                      v65 = 1;
                      v46 = v106 + 1;
                      v37 = (__int64 *)(v105 + 16LL * v64);
                      v66 = *v37;
                      if ( *v37 != v44 )
                      {
                        while ( v66 != -4096 )
                        {
                          if ( !v63 && v66 == -8192 )
                            v63 = v37;
                          v64 = (v107 - 1) & (v65 + v64);
                          v37 = (__int64 *)(v105 + 16LL * v64);
                          v66 = *v37;
                          if ( v44 == *v37 )
                            goto LABEL_71;
                          ++v65;
                        }
                        if ( v63 )
                          v37 = v63;
                      }
                    }
                    goto LABEL_71;
                  }
LABEL_69:
                  sub_2513E70((__int64)&v104, 2 * v107);
                  if ( !v107 )
                    goto LABEL_189;
                  v45 = (v107 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                  v46 = v106 + 1;
                  v37 = (__int64 *)(v105 + 16LL * v45);
                  v47 = *v37;
                  if ( *v37 != v44 )
                  {
                    v93 = 1;
                    v94 = 0;
                    while ( v47 != -4096 )
                    {
                      if ( !v94 && v47 == -8192 )
                        v94 = v37;
                      v45 = (v107 - 1) & (v93 + v45);
                      v37 = (__int64 *)(v105 + 16LL * v45);
                      v47 = *v37;
                      if ( v44 == *v37 )
                        goto LABEL_71;
                      ++v93;
                    }
                    if ( v94 )
                      v37 = v94;
                  }
LABEL_71:
                  LODWORD(v106) = v46;
                  if ( *v37 != -4096 )
                    --HIDWORD(v106);
                  *((_DWORD *)v37 + 2) = 0;
                  *v37 = v44;
                  v41 = v37 + 1;
                  goto LABEL_74;
                }
LABEL_63:
                v41 = v39 + 1;
                if ( *((_BYTE *)v39 + 10) )
                {
                  v42 = *((_WORD *)v39 + 4);
LABEL_65:
                  v43 = v42 - 1;
                  *(_WORD *)v41 = v43;
                  LODWORD(i) = v109;
                  if ( !v43 )
                    goto LABEL_76;
                  goto LABEL_66;
                }
LABEL_74:
                v42 = sub_BD3960(v102);
                v48 = *((_BYTE *)v41 + 2) == 0;
                *(_WORD *)v41 = v42;
                if ( !v48 )
                  goto LABEL_65;
                v49 = v42 - 1;
                *((_BYTE *)v41 + 2) = 1;
                *(_WORD *)v41 = v49;
                LODWORD(i) = v109;
                if ( !v49 )
                {
LABEL_76:
                  v50 = *(_DWORD *)(a1 + 184);
                  if ( v50 )
                  {
                    v51 = v102;
                    v52 = v50 - 1;
                    v53 = 1;
                    v54 = 0;
                    v55 = *(_QWORD *)(a1 + 168);
                    v56 = v52 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
                    v57 = (__int64 *)(v55 + 8LL * v56);
                    v58 = *v57;
                    if ( v102 == *v57 )
                    {
LABEL_78:
                      v59 = 4LL * (*(_DWORD *)(v51 + 4) & 0x7FFFFFF);
                      if ( (*(_BYTE *)(v51 + 7) & 0x40) != 0 )
                      {
                        v60 = *(_QWORD **)(v51 - 8);
                        v61 = &v60[v59];
                      }
                      else
                      {
                        v61 = (_QWORD *)v51;
                        v60 = (_QWORD *)(v51 - v59 * 8);
                      }
                      for ( i = (unsigned int)v109; v61 != v60; v60 += 4 )
                      {
                        v62 = (_BYTE *)*v60;
                        if ( *(_BYTE *)*v60 > 0x1Cu )
                        {
                          if ( i + 1 > (unsigned __int64)HIDWORD(v109) )
                          {
                            sub_C8D5F0((__int64)&v108, v110, i + 1, 8u, i + 1, v52);
                            i = (unsigned int)v109;
                          }
                          v108[i] = (__int64)v62;
                          i = (unsigned int)(v109 + 1);
                          LODWORD(v109) = v109 + 1;
                        }
                      }
                      goto LABEL_66;
                    }
                    while ( v58 != -4096 )
                    {
                      if ( v58 != -8192 || v54 )
                        v57 = v54;
                      v56 = v52 & (v53 + v56);
                      v67 = (__int64 *)(v55 + 8LL * v56);
                      v58 = *v67;
                      if ( v102 == *v67 )
                        goto LABEL_78;
                      ++v53;
                      v54 = v57;
                      v57 = (__int64 *)(v55 + 8LL * v56);
                    }
                    if ( !v54 )
                      v54 = v57;
                    v68 = *(_DWORD *)(a1 + 176);
                    ++*(_QWORD *)(a1 + 160);
                    v69 = v68 + 1;
                    v103 = v54;
                    if ( 4 * v69 < 3 * v50 )
                    {
                      if ( v50 - *(_DWORD *)(a1 + 180) - v69 > v50 >> 3 )
                      {
LABEL_115:
                        *(_DWORD *)(a1 + 176) = v69;
                        if ( *v54 != -4096 )
                          --*(_DWORD *)(a1 + 180);
                        *v54 = v51;
                        v70 = *(unsigned int *)(a1 + 200);
                        v71 = v102;
                        if ( v70 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 204) )
                        {
                          sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v70 + 1, 8u, v55, v52);
                          v70 = *(unsigned int *)(a1 + 200);
                        }
                        *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v70) = v71;
                        v51 = v102;
                        ++*(_DWORD *)(a1 + 200);
                        goto LABEL_78;
                      }
LABEL_122:
                      sub_22EE7D0(v95, v50);
                      sub_2512810(v95, &v102, &v103);
                      v51 = v102;
                      v54 = v103;
                      v69 = *(_DWORD *)(a1 + 176) + 1;
                      goto LABEL_115;
                    }
                  }
                  else
                  {
                    ++*(_QWORD *)(a1 + 160);
                    v103 = 0;
                  }
                  v50 *= 2;
                  goto LABEL_122;
                }
LABEL_66:
                v33 = v108;
                if ( !(_DWORD)i )
                {
                  v3 = v97;
                  j = v96;
                  if ( v108 != v110 )
                    _libc_free((unsigned __int64)v108);
                  goto LABEL_19;
                }
              }
              ++v104;
              goto LABEL_69;
            }
            v86 = 1;
            v87 = 0;
            while ( v32 != -4096 )
            {
              if ( !v87 && v32 == -8192 )
                v87 = v31;
              v30 = v28 & (v86 + v30);
              v31 = (__int64 *)(v29 + 8LL * v30);
              v32 = *v31;
              if ( v101 == *v31 )
                goto LABEL_60;
              ++v86;
            }
            if ( !v87 )
              v87 = v31;
            ++*(_QWORD *)(a1 + 160);
            v88 = *(_DWORD *)(a1 + 176);
            v108 = v87;
            v89 = v88 + 1;
            if ( 4 * (v88 + 1) < 3 * v27 )
            {
              v90 = j - 24;
              if ( v27 - *(_DWORD *)(a1 + 180) - v89 > v27 >> 3 )
              {
LABEL_143:
                *(_DWORD *)(a1 + 176) = v89;
                if ( *v87 != -4096 )
                  --*(_DWORD *)(a1 + 180);
                *v87 = v90;
                v91 = (__int64)v103;
                v92 = *(unsigned int *)(a1 + 200);
                if ( v92 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 204) )
                {
                  sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v92 + 1, 8u, v28, (__int64)v87);
                  v92 = *(unsigned int *)(a1 + 200);
                }
                *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v92) = v91;
                ++*(_DWORD *)(a1 + 200);
                goto LABEL_60;
              }
LABEL_150:
              sub_22EE7D0(v95, v27);
              sub_2512810(v95, (__int64 *)&v103, &v108);
              v90 = (__int64)v103;
              v87 = v108;
              v89 = *(_DWORD *)(a1 + 176) + 1;
              goto LABEL_143;
            }
          }
          else
          {
            v108 = 0;
            ++*(_QWORD *)(a1 + 160);
          }
          v27 *= 2;
          goto LABEL_150;
        }
        if ( (*(_WORD *)(j - 22) & 3) != 2 )
          goto LABEL_20;
        *(_BYTE *)(a3 + 113) = 1;
        v9 = *(_BYTE **)(j - 56);
        if ( v9 && !*v9 )
          *(_BYTE *)(sub_251B1C0(a1) + 112) = 1;
LABEL_19:
        v7 = *(unsigned __int8 *)(j - 24) - 29;
LABEL_20:
        v10 = *(_DWORD *)(a3 + 24);
        if ( !v10 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_124;
        }
        v11 = *(_QWORD *)(a3 + 8);
        v12 = v10 - 1;
        v13 = 0;
        v14 = 1;
        v15 = v12 & (37 * v7);
        v16 = v11 + 16LL * v15;
        v17 = *(_DWORD *)v16;
        if ( v7 == *(_DWORD *)v16 )
          goto LABEL_22;
        break;
      default:
        goto LABEL_26;
    }
    while ( 1 )
    {
      if ( v17 == -1 )
      {
        if ( !v13 )
          v13 = v16;
        ++*(_QWORD *)a3;
        v25 = *(_DWORD *)(a3 + 16) + 1;
        if ( 4 * v25 >= 3 * v10 )
        {
LABEL_124:
          sub_2514050(a3, 2 * v10);
          v72 = *(_DWORD *)(a3 + 24);
          if ( v72 )
          {
            v73 = v72 - 1;
            v74 = *(_QWORD *)(a3 + 8);
            LODWORD(v75) = v73 & (37 * v7);
            v25 = *(_DWORD *)(a3 + 16) + 1;
            v13 = v74 + 16LL * (unsigned int)v75;
            v76 = *(_DWORD *)v13;
            if ( v7 != *(_DWORD *)v13 )
            {
              v77 = 1;
              v78 = 0;
              while ( v76 != -1 )
              {
                if ( !v78 && v76 == -2 )
                  v78 = v13;
                v75 = v73 & (unsigned int)(v75 + v77);
                v13 = v74 + 16 * v75;
                v76 = *(_DWORD *)v13;
                if ( v7 == *(_DWORD *)v13 )
                  goto LABEL_47;
                ++v77;
              }
              if ( v78 )
                v13 = v78;
            }
            goto LABEL_47;
          }
        }
        else
        {
          if ( v10 - *(_DWORD *)(a3 + 20) - v25 > v10 >> 3 )
          {
LABEL_47:
            *(_DWORD *)(a3 + 16) = v25;
            if ( *(_DWORD *)v13 != -1 )
              --*(_DWORD *)(a3 + 20);
            *(_DWORD *)v13 = v7;
            v19 = (_QWORD *)(v13 + 8);
            *(_QWORD *)(v13 + 8) = 0;
LABEL_50:
            v26 = (_QWORD *)sub_A777F0(0x50u, *(__int64 **)(a1 + 112));
            v18 = (__int64)v26;
            if ( v26 )
            {
              *v26 = v26 + 2;
              v26[1] = 0x800000000LL;
            }
            *v19 = v26;
            goto LABEL_23;
          }
          sub_2514050(a3, v10);
          v79 = *(_DWORD *)(a3 + 24);
          if ( v79 )
          {
            v80 = v79 - 1;
            v81 = *(_QWORD *)(a3 + 8);
            v82 = 0;
            LODWORD(v83) = v80 & (37 * v7);
            v84 = 1;
            v25 = *(_DWORD *)(a3 + 16) + 1;
            v13 = v81 + 16LL * (unsigned int)v83;
            v85 = *(_DWORD *)v13;
            if ( v7 != *(_DWORD *)v13 )
            {
              while ( v85 != -1 )
              {
                if ( v85 == -2 && !v82 )
                  v82 = v13;
                v83 = v80 & (unsigned int)(v83 + v84);
                v13 = v81 + 16 * v83;
                v85 = *(_DWORD *)v13;
                if ( v7 == *(_DWORD *)v13 )
                  goto LABEL_47;
                ++v84;
              }
              if ( v82 )
                v13 = v82;
            }
            goto LABEL_47;
          }
        }
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
      if ( v13 || v17 != -2 )
        v16 = v13;
      v15 = v12 & (v14 + v15);
      v17 = *(_DWORD *)(v11 + 16LL * v15);
      if ( v7 == v17 )
        break;
      ++v14;
      v13 = v16;
      v16 = v11 + 16LL * v15;
    }
    v16 = v11 + 16LL * v15;
LABEL_22:
    v18 = *(_QWORD *)(v16 + 8);
    v19 = (_QWORD *)(v16 + 8);
    if ( !v18 )
      goto LABEL_50;
LABEL_23:
    v20 = *(unsigned int *)(v18 + 8);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
    {
      sub_C8D5F0(v18, (const void *)(v18 + 16), v20 + 1, 8u, v11, v12);
      v20 = *(unsigned int *)(v18 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v18 + 8 * v20) = v101;
    ++*(_DWORD *)(v18 + 8);
LABEL_26:
    if ( (unsigned __int8)sub_B46420(v101) || (unsigned __int8)sub_B46490(v101) )
    {
      v23 = *(unsigned int *)(a3 + 40);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
      {
        sub_C8D5F0(a3 + 32, (const void *)(a3 + 48), v23 + 1, 8u, v21, v22);
        v23 = *(unsigned int *)(a3 + 40);
      }
      *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * v23) = v101;
      ++*(_DWORD *)(a3 + 40);
    }
    for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v4 + 32) )
    {
      v24 = v4 - 24;
      if ( !v4 )
        v24 = 0;
      if ( j != v24 + 48 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        goto LABEL_7;
      if ( !v4 )
        BUG();
    }
  }
  while ( v3 != v4 );
LABEL_7:
  if ( (unsigned __int8)sub_B2D610(a2, 3) && !sub_30D6380(a2) )
    sub_AE6EC0(a1 + 248, a2);
  return sub_C7D6A0(v105, 16LL * v107, 8);
}
