// Function: sub_24AEB60
// Address: 0x24aeb60
//
__int64 __fastcall sub_24AEB60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  unsigned __int64 v15; // r14
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r10
  _BYTE *v24; // rsi
  void *v25; // r15
  __int64 v26; // r8
  __int64 **v27; // rbx
  __int64 **i; // r15
  __int64 *v29; // r14
  __int64 v30; // rsi
  __int64 v31; // r10
  __int64 v32; // rcx
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // r12
  __int64 v37; // r12
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r11
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 *v44; // r12
  unsigned int v45; // r12d
  char *v47; // r15
  char *v48; // rcx
  signed __int64 v49; // r13
  __int64 v50; // rax
  char *v51; // rbx
  char *v52; // r14
  char *v53; // r13
  char *v54; // r8
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rcx
  bool v58; // cf
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // r12
  char *v61; // rcx
  char *v62; // r15
  __int64 v63; // rax
  _QWORD *v64; // rbx
  _BYTE *v65; // rsi
  __int64 v66; // r13
  __int64 v67; // r14
  unsigned __int64 v68; // rax
  int v69; // edx
  _BYTE *v70; // r12
  _BYTE *v71; // rax
  unsigned int v72; // esi
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 *v76; // r12
  __int64 *v77; // r13
  __int64 v78; // rbx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // rsi
  int v82; // edi
  unsigned int v83; // edx
  __int64 *v84; // rax
  __int64 v85; // r10
  __int64 v86; // rax
  __int64 v87; // r9
  unsigned int v88; // edx
  __int64 *v89; // rax
  __int64 v90; // r11
  char v91; // dl
  __int64 v92; // rsi
  __int64 v93; // rcx
  __int64 v94; // rdi
  unsigned int v95; // edx
  __int64 *v96; // rax
  __int64 v97; // r10
  __int64 v98; // rax
  char *v99; // rax
  int v100; // eax
  int v101; // eax
  __int64 v102; // rax
  int v103; // eax
  int v104; // eax
  int v105; // eax
  int v106; // eax
  int v107; // r9d
  int v108; // r11d
  int v109; // r10d
  int v110; // r9d
  int v111; // r10d
  unsigned __int64 v112; // r12
  __int64 v113; // rax
  unsigned __int64 v114; // [rsp+8h] [rbp-E8h]
  char *v115; // [rsp+18h] [rbp-D8h]
  char *v116; // [rsp+18h] [rbp-D8h]
  int v117; // [rsp+20h] [rbp-D0h]
  __int64 v118; // [rsp+20h] [rbp-D0h]
  char *v119; // [rsp+28h] [rbp-C8h]
  char *v120; // [rsp+28h] [rbp-C8h]
  char *v121; // [rsp+28h] [rbp-C8h]
  __int64 v122; // [rsp+28h] [rbp-C8h]
  __int64 v123; // [rsp+28h] [rbp-C8h]
  __int64 v124; // [rsp+28h] [rbp-C8h]
  _BYTE *v127; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE *v128; // [rsp+48h] [rbp-A8h]
  _BYTE *v129; // [rsp+50h] [rbp-A0h]
  _QWORD v130[4]; // [rsp+60h] [rbp-90h] BYREF
  int v131; // [rsp+80h] [rbp-70h]
  char v132; // [rsp+84h] [rbp-6Ch]
  void *v133[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v134; // [rsp+A0h] [rbp-50h]
  __int16 v135; // [rsp+B0h] [rbp-40h]

  v6 = *(_BYTE *)(a1 + 424) == 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  if ( v6 )
  {
    v47 = *(char **)(a1 + 256);
    v48 = *(char **)(a1 + 248);
    v49 = v47 - v48;
    if ( (unsigned __int64)(v47 - v48) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"vector::reserve");
    if ( v49 )
    {
      v50 = sub_22077B0(*(_QWORD *)(a1 + 256) - (_QWORD)v48);
      v51 = (char *)v50;
      v48 = *(char **)(a1 + 248);
      v47 = *(char **)(a1 + 256);
      v115 = (char *)(v50 + v49);
      if ( v47 == v48 )
      {
        v114 = v50;
LABEL_111:
        if ( v114 )
          j_j___libc_free_0(v114);
LABEL_25:
        if ( !*(_BYTE *)(a1 + 424) )
        {
          v26 = a1;
          v27 = *(__int64 ***)(a1 + 248);
          for ( i = *(__int64 ***)(a1 + 256); i != v27; ++v27 )
          {
            v29 = *v27;
            if ( !*((_BYTE *)*v27 + 25) )
            {
              v30 = *(unsigned int *)(v26 + 296);
              v31 = *v29;
              v32 = v29[1];
              v33 = *(_QWORD *)(v26 + 280);
              if ( (_DWORD)v30 )
              {
                a6 = (unsigned int)(v30 - 1);
                v34 = a6 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                v35 = (__int64 *)(v33 + 16LL * v34);
                v36 = *v35;
                if ( v31 == *v35 )
                {
LABEL_30:
                  v37 = v35[1];
                }
                else
                {
                  v103 = 1;
                  while ( v36 != -4096 )
                  {
                    v108 = v103 + 1;
                    v34 = a6 & (v103 + v34);
                    v35 = (__int64 *)(v33 + 16LL * v34);
                    v36 = *v35;
                    if ( v31 == *v35 )
                      goto LABEL_30;
                    v103 = v108;
                  }
                  v37 = *(_QWORD *)(v33 + 16LL * (unsigned int)v30 + 8);
                }
                v38 = a6 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
                v39 = (__int64 *)(v33 + 16LL * v38);
                v40 = *v39;
                if ( v32 == *v39 )
                {
LABEL_32:
                  v41 = v39[1];
                }
                else
                {
                  v104 = 1;
                  while ( v40 != -4096 )
                  {
                    v109 = v104 + 1;
                    v38 = a6 & (v104 + v38);
                    v39 = (__int64 *)(v33 + 16LL * v38);
                    v40 = *v39;
                    if ( v32 == *v39 )
                      goto LABEL_32;
                    v104 = v109;
                  }
                  v41 = *(_QWORD *)(v33 + 16 * v30 + 8);
                }
              }
              else
              {
                v37 = *(_QWORD *)(v33 + 8);
                v41 = v37;
              }
              v42 = *(unsigned int *)(v37 + 80);
              if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 84) )
              {
                v122 = v26;
                sub_C8D5F0(v37 + 72, (const void *)(v37 + 88), v42 + 1, 8u, v26, a6);
                v42 = *(unsigned int *)(v37 + 80);
                v26 = v122;
              }
              *(_QWORD *)(*(_QWORD *)(v37 + 72) + 8 * v42) = v29;
              ++*(_DWORD *)(v37 + 80);
              ++*(_DWORD *)(v37 + 36);
              v43 = *(unsigned int *)(v41 + 48);
              v44 = *v27;
              if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 52) )
              {
                v123 = v26;
                sub_C8D5F0(v41 + 40, (const void *)(v41 + 56), v43 + 1, 8u, v26, a6);
                v43 = *(unsigned int *)(v41 + 48);
                v26 = v123;
              }
              *(_QWORD *)(*(_QWORD *)(v41 + 40) + 8 * v43) = v44;
              ++*(_DWORD *)(v41 + 48);
              ++*(_DWORD *)(v41 + 32);
            }
          }
        }
        goto LABEL_3;
      }
    }
    else
    {
      v115 = 0;
      v51 = 0;
      if ( v47 == v48 )
        goto LABEL_25;
    }
    v52 = v115;
    v53 = v48;
    v54 = v51;
    do
    {
      while ( 1 )
      {
        v55 = *(_QWORD *)v53;
        if ( v52 != v51 )
          break;
        a6 = v52 - v54;
        v56 = (v52 - v54) >> 3;
        if ( v56 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v57 = 1;
        if ( v56 )
          v57 = (v52 - v54) >> 3;
        v58 = __CFADD__(v57, v56);
        v59 = v57 + v56;
        if ( v58 )
        {
          v112 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v59 )
          {
            v60 = 0;
            v61 = 0;
            goto LABEL_58;
          }
          if ( v59 > 0xFFFFFFFFFFFFFFFLL )
            v59 = 0xFFFFFFFFFFFFFFFLL;
          v112 = 8 * v59;
        }
        v116 = v54;
        v118 = v52 - v54;
        v124 = *(_QWORD *)v53;
        v113 = sub_22077B0(v112);
        v55 = v124;
        a6 = v118;
        v54 = v116;
        v61 = (char *)v113;
        v60 = v113 + v112;
LABEL_58:
        if ( &v61[a6] )
          *(_QWORD *)&v61[a6] = v55;
        v51 = &v61[a6 + 8];
        if ( a6 > 0 )
        {
          v120 = v54;
          v99 = (char *)memmove(v61, v54, a6);
          v54 = v120;
          v61 = v99;
LABEL_117:
          v121 = v61;
          j_j___libc_free_0((unsigned __int64)v54);
          v61 = v121;
          goto LABEL_62;
        }
        if ( v54 )
          goto LABEL_117;
LABEL_62:
        v53 += 8;
        v52 = (char *)v60;
        v54 = v61;
        if ( v47 == v53 )
          goto LABEL_63;
      }
      if ( v51 )
        *(_QWORD *)v51 = v55;
      v53 += 8;
      v51 += 8;
    }
    while ( v47 != v53 );
LABEL_63:
    v114 = (unsigned __int64)v54;
    if ( v51 != v54 )
    {
      v119 = v51;
      v62 = v54;
      while ( 1 )
      {
        v66 = *(_QWORD *)v62;
        if ( *(_BYTE *)(*(_QWORD *)v62 + 24LL) || *(_BYTE *)(v66 + 25) )
          goto LABEL_72;
        v64 = *(_QWORD **)v66;
        v67 = *(_QWORD *)(v66 + 8);
        if ( *(_QWORD *)v66 )
        {
          if ( !v67 )
            goto LABEL_68;
          v68 = v64[6] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v64 + 6 == (_QWORD *)v68 )
          {
            v70 = 0;
          }
          else
          {
            if ( !v68 )
              BUG();
            v69 = *(unsigned __int8 *)(v68 - 24);
            v70 = 0;
            v71 = (_BYTE *)(v68 - 24);
            if ( (unsigned int)(v69 - 30) < 0xB )
              v70 = v71;
          }
          if ( (unsigned int)sub_B46E30((__int64)v70) <= 1 )
          {
            v98 = sub_AA5190((__int64)v64);
            if ( !v98 || v64 + 6 != (_QWORD *)v98 )
              goto LABEL_68;
          }
          else if ( *(_BYTE *)(v66 + 26) )
          {
            v72 = sub_D0E820((__int64)v64, v67);
            if ( *v70 != 33 )
            {
              v135 = 257;
              memset(v130, 0, sizeof(v130));
              v131 = 0;
              v132 = 1;
              v73 = sub_F451F0((__int64)v70, v72, (__int64)v130, v133);
              if ( v73 )
              {
                sub_24A4AF0(a1 + 240, (__int64)v64, v73, 0);
                *(_BYTE *)(sub_24A4AF0(a1 + 240, v73, v67, 0) + 24) = 1;
                *(_BYTE *)(v66 + 25) = 1;
                v74 = sub_AA5190(v73);
                if ( !v74 || v74 != v73 + 48 )
                {
                  v64 = (_QWORD *)v73;
LABEL_68:
                  v133[0] = v64;
                  v65 = v128;
                  if ( v128 != v129 )
                    goto LABEL_69;
                  goto LABEL_110;
                }
              }
            }
          }
          else
          {
            v63 = sub_AA5190(v67);
            if ( !v63 || v63 != v67 + 48 )
            {
              v64 = (_QWORD *)v67;
              goto LABEL_68;
            }
          }
LABEL_72:
          v62 += 8;
          if ( v119 == v62 )
            goto LABEL_111;
        }
        else
        {
          v133[0] = *(void **)(v66 + 8);
          if ( !v67 )
            goto LABEL_72;
          v65 = v128;
          v64 = (_QWORD *)v67;
          if ( v128 != v129 )
          {
LABEL_69:
            if ( v65 )
            {
              *(_QWORD *)v65 = v64;
              v65 = v128;
            }
            v128 = v65 + 8;
            goto LABEL_72;
          }
LABEL_110:
          v62 += 8;
          sub_9319A0((__int64)&v127, v65, v133);
          if ( v119 == v62 )
            goto LABEL_111;
        }
      }
    }
    goto LABEL_111;
  }
  v7 = *(_QWORD *)(a1 + 32);
  v8 = *(_QWORD *)(v7 + 80);
  v9 = v7 + 72;
  if ( v8 != v9 )
  {
    do
    {
      while ( 1 )
      {
        v25 = (void *)(v8 - 24);
        if ( !v8 )
          v25 = 0;
        if ( (unsigned __int8)sub_3158140(a1 + 344, v25) )
          break;
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          goto LABEL_25;
      }
      v133[0] = v25;
      v24 = v128;
      if ( v128 == v129 )
      {
        sub_F38A10((__int64)&v127, v128, v133);
      }
      else
      {
        if ( v128 )
        {
          *(_QWORD *)v128 = v25;
          v24 = v128;
        }
        v128 = v24 + 8;
      }
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v9 != v8 );
    goto LABEL_25;
  }
LABEL_3:
  v10 = (unsigned __int64)v127;
  v11 = *a2;
  v12 = *(_DWORD *)(a1 + 104) + (unsigned int)((v128 - v127) >> 3);
  if ( v12 != (a2[1] - *a2) >> 3 )
  {
    v45 = 0;
    goto LABEL_41;
  }
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( v13 )
    v13 -= 24;
  if ( v127 == v128 )
  {
    v76 = *(__int64 **)(a1 + 248);
    *(_DWORD *)(a1 + 444) = v12;
    v77 = *(__int64 **)(a1 + 256);
    *(_DWORD *)(a1 + 440) = 0;
    if ( v77 != v76 )
      goto LABEL_90;
    v45 = 1;
LABEL_41:
    if ( v10 )
      goto LABEL_42;
    return v45;
  }
  v14 = 0;
  v15 = (unsigned __int64)(v128 - 8 - v127) >> 3;
  while ( 1 )
  {
    v21 = *(unsigned int *)(a1 + 296);
    v22 = *(_QWORD *)(v10 + 8 * v14);
    v20 = *(_QWORD *)(v11 + 8LL * (unsigned int)v14);
    v23 = *(_QWORD *)(a1 + 280);
    if ( !(_DWORD)v21 )
      goto LABEL_15;
    v16 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v17 = (__int64 *)(v23 + 16LL * v16);
    v18 = *v17;
    if ( v22 != *v17 )
    {
      v101 = 1;
      while ( v18 != -4096 )
      {
        v16 = (v21 - 1) & (v101 + v16);
        v117 = v101 + 1;
        v17 = (__int64 *)(v23 + 16LL * v16);
        v18 = *v17;
        if ( v22 == *v17 )
          goto LABEL_9;
        v101 = v117;
      }
LABEL_15:
      v17 = (__int64 *)(v23 + 16 * v21);
    }
LABEL_9:
    v19 = v17[1];
    if ( v13 == v22 && !v20 )
      v20 = 1;
    *(_QWORD *)(v19 + 16) = v20;
    *(_BYTE *)(v19 + 24) = 1;
    if ( v15 == v14 )
      break;
    ++v14;
    v11 = *a2;
  }
  v75 = (a2[1] - *a2) >> 3;
  *(_DWORD *)(a1 + 440) = v14 + 1;
  v76 = *(__int64 **)(a1 + 248);
  v77 = *(__int64 **)(a1 + 256);
  *(_DWORD *)(a1 + 444) = v75;
  if ( v77 != v76 )
  {
    while ( 1 )
    {
LABEL_90:
      v78 = *v76;
      if ( *(_BYTE *)(*v76 + 25) || *(_BYTE *)(v78 + 24) )
        goto LABEL_106;
      v79 = *(unsigned int *)(a1 + 296);
      v80 = *(_QWORD *)v78;
      v81 = *(_QWORD *)(a1 + 280);
      if ( (_DWORD)v79 )
      {
        v82 = v79 - 1;
        v83 = (v79 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
        v84 = (__int64 *)(v81 + 16LL * v83);
        v85 = *v84;
        if ( v80 == *v84 )
        {
LABEL_94:
          v86 = v84[1];
          if ( !*(_BYTE *)(v86 + 24) )
            goto LABEL_121;
        }
        else
        {
          v100 = 1;
          while ( v85 != -4096 )
          {
            v110 = v100 + 1;
            v83 = v82 & (v100 + v83);
            v84 = (__int64 *)(v81 + 16LL * v83);
            v85 = *v84;
            if ( v80 == *v84 )
              goto LABEL_94;
            v100 = v110;
          }
          v86 = *(_QWORD *)(v81 + 16LL * (unsigned int)v79 + 8);
          if ( !*(_BYTE *)(v86 + 24) )
          {
LABEL_121:
            v87 = *(_QWORD *)(v78 + 8);
            goto LABEL_98;
          }
        }
      }
      else
      {
        v86 = *(_QWORD *)(v81 + 8);
        if ( !*(_BYTE *)(v86 + 24) )
          goto LABEL_102;
      }
      if ( *(_DWORD *)(v86 + 80) == 1 )
        goto LABEL_126;
      v87 = *(_QWORD *)(v78 + 8);
      if ( !(_DWORD)v79 )
      {
        v86 = *(_QWORD *)(v81 + 8);
        v91 = *(_BYTE *)(v86 + 24);
        goto LABEL_100;
      }
      v82 = v79 - 1;
LABEL_98:
      v88 = v82 & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
      v89 = (__int64 *)(v81 + 16LL * v88);
      v90 = *v89;
      if ( v87 == *v89 )
      {
LABEL_99:
        v86 = v89[1];
        v91 = *(_BYTE *)(v86 + 24);
      }
      else
      {
        v106 = 1;
        while ( v90 != -4096 )
        {
          v111 = v106 + 1;
          v88 = v82 & (v106 + v88);
          v89 = (__int64 *)(v81 + 16LL * v88);
          v90 = *v89;
          if ( v87 == *v89 )
            goto LABEL_99;
          v106 = v111;
        }
        v86 = *(_QWORD *)(v81 + 16 * v79 + 8);
        v91 = *(_BYTE *)(v86 + 24);
      }
LABEL_100:
      if ( v91 && *(_DWORD *)(v86 + 48) == 1 )
      {
LABEL_126:
        v102 = *(_QWORD *)(v86 + 16);
        *(_BYTE *)(v78 + 40) = 1;
        *(_QWORD *)(v78 + 32) = v102;
        sub_24A2C40((__int64 **)v133, (__int64 *)(a1 + 272), v80);
        --*(_DWORD *)(*(_QWORD *)(v134 + 8) + 36LL);
        sub_24A2C40((__int64 **)v133, (__int64 *)(a1 + 272), *(_QWORD *)(v78 + 8));
        --*(_DWORD *)(*(_QWORD *)(v134 + 8) + 32LL);
        v78 = *v76;
      }
LABEL_102:
      if ( !*(_BYTE *)(v78 + 40) )
      {
        *(_QWORD *)(v78 + 32) = 0;
        v92 = *(_QWORD *)v78;
        *(_BYTE *)(v78 + 40) = 1;
        v93 = *(unsigned int *)(a1 + 296);
        v94 = *(_QWORD *)(a1 + 280);
        if ( (_DWORD)v93 )
        {
          v95 = (v93 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
          v96 = (__int64 *)(v94 + 16LL * v95);
          v97 = *v96;
          if ( v92 == *v96 )
          {
LABEL_105:
            --*(_DWORD *)(v96[1] + 36);
            sub_24A2C40((__int64 **)v133, (__int64 *)(a1 + 272), *(_QWORD *)(v78 + 8));
            --*(_DWORD *)(*(_QWORD *)(v134 + 8) + 32LL);
            goto LABEL_106;
          }
          v105 = 1;
          while ( v97 != -4096 )
          {
            v107 = v105 + 1;
            v95 = (v93 - 1) & (v105 + v95);
            v96 = (__int64 *)(v94 + 16LL * v95);
            v97 = *v96;
            if ( v92 == *v96 )
              goto LABEL_105;
            v105 = v107;
          }
        }
        v96 = (__int64 *)(v94 + 16 * v93);
        goto LABEL_105;
      }
LABEL_106:
      if ( v77 == ++v76 )
      {
        v10 = (unsigned __int64)v127;
        v45 = 1;
        goto LABEL_41;
      }
    }
  }
  v45 = 1;
LABEL_42:
  j_j___libc_free_0(v10);
  return v45;
}
