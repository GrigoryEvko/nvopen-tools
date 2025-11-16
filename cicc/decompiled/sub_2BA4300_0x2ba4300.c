// Function: sub_2BA4300
// Address: 0x2ba4300
//
void __fastcall sub_2BA4300(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r9
  _QWORD *v5; // rbx
  int v6; // edi
  __int64 v7; // rbx
  __int64 v8; // rcx
  int v9; // esi
  __int64 v10; // r8
  int v11; // esi
  unsigned int v12; // edx
  __int64 *v13; // rax
  _QWORD *v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rcx
  _QWORD *v17; // rbx
  __int64 v18; // rbx
  __int64 v19; // r14
  int v20; // ecx
  __int64 v21; // rsi
  int v22; // ecx
  unsigned int v23; // edx
  __int64 *v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // esi
  int v29; // ecx
  _QWORD *v30; // rdx
  _QWORD *v31; // rsi
  __int64 v32; // rax
  __int64 *v33; // r14
  __int64 v34; // r13
  int *v35; // rax
  _QWORD *v36; // r15
  __int64 *v37; // rbx
  _QWORD *v38; // r12
  __int64 v39; // rsi
  __int64 v40; // r8
  __int64 v41; // r15
  _QWORD *v42; // r9
  __int64 v43; // rax
  __int64 v44; // rbx
  _QWORD *v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // r9
  __int64 v50; // rcx
  __int64 v51; // r14
  __int64 v52; // r15
  __int64 v53; // r8
  __int64 v54; // r13
  __int64 v55; // rax
  int v56; // edi
  __int64 v57; // r9
  int v58; // edi
  unsigned int v59; // esi
  __int64 *v60; // rdx
  __int64 v61; // r11
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rax
  int v65; // esi
  int v66; // edx
  _QWORD *v67; // rax
  _QWORD *v68; // rdx
  __int64 *v69; // rbx
  __int64 *v70; // r15
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rax
  int v74; // ecx
  int v75; // edx
  _QWORD *v76; // rdx
  _QWORD *v77; // rsi
  __int64 v78; // rbx
  __int64 v79; // r15
  __int64 v80; // rsi
  __int64 v81; // rax
  int v82; // ecx
  int v83; // edx
  _QWORD *v84; // rdx
  _QWORD *v85; // rsi
  __int64 v86; // rax
  _QWORD *v87; // rbx
  _BYTE *v88; // rax
  int v89; // esi
  __int64 v90; // rdi
  int v91; // esi
  unsigned int v92; // ecx
  _QWORD *v93; // rdx
  _BYTE *v94; // r8
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rax
  int v98; // ecx
  int v99; // edx
  _QWORD *v100; // rdx
  _QWORD *v101; // rsi
  int v102; // edx
  int v103; // r10d
  int v104; // edx
  int v105; // r9d
  int v106; // eax
  int v107; // r8d
  int v108; // eax
  _QWORD *v109; // [rsp-A0h] [rbp-A0h]
  __int64 v110; // [rsp-90h] [rbp-90h]
  __int64 v111; // [rsp-88h] [rbp-88h]
  __int64 v112; // [rsp-80h] [rbp-80h]
  __int64 v113; // [rsp-78h] [rbp-78h] BYREF
  __int64 v114; // [rsp-70h] [rbp-70h] BYREF
  __int64 v115; // [rsp-68h] [rbp-68h] BYREF
  __int64 v116; // [rsp-60h] [rbp-60h] BYREF
  unsigned __int64 v117; // [rsp-58h] [rbp-58h]
  __int64 *v118; // [rsp-50h] [rbp-50h]
  __int64 *v119; // [rsp-48h] [rbp-48h]
  __int64 v120; // [rsp-40h] [rbp-40h]

  if ( *(_QWORD *)(a2 + 160) )
  {
    v3 = a2;
    sub_2B2F0D0(a2);
    LODWORD(v116) = 0;
    v5 = *(_QWORD **)(a2 + 160);
    v117 = 0;
    v118 = &v116;
    v119 = &v116;
    v120 = 0;
    if ( v5 == *(_QWORD **)(a2 + 168) )
      goto LABEL_87;
    v6 = 0;
    while ( 1 )
    {
      v8 = v5[5];
      if ( *(_QWORD *)v3 != v8 )
        goto LABEL_4;
      v9 = *(_DWORD *)(v3 + 104);
      v10 = *(_QWORD *)(v3 + 88);
      if ( !v9 )
        goto LABEL_4;
      v11 = v9 - 1;
      v12 = v11 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = (_QWORD *)*v13;
      if ( v5 == (_QWORD *)*v13 )
      {
LABEL_10:
        v15 = v13[1];
        if ( !v15 || *(_DWORD *)(v15 + 136) != *(_DWORD *)(v3 + 204) )
          goto LABEL_4;
        *(_DWORD *)(*(_QWORD *)(v15 + 16) + 140LL) = v6;
        if ( *(_QWORD *)(v15 + 16) == v15 && (*(_QWORD *)(v15 + 24) || *(_QWORD *)(v15 + 8)) )
          sub_2BA36A0(v3, v15, 0, a1, v10, v4);
        v16 = v5[5];
        v7 = v5[4];
        ++v6;
        if ( v7 == v16 + 48 )
          goto LABEL_14;
LABEL_5:
        if ( !v7 )
          goto LABEL_14;
        v5 = (_QWORD *)(v7 - 24);
        if ( *(_QWORD **)(v3 + 168) == v5 )
          goto LABEL_15;
      }
      else
      {
        v108 = 1;
        while ( v14 != (_QWORD *)-4096LL )
        {
          v4 = (unsigned int)(v108 + 1);
          v12 = v11 & (v108 + v12);
          v13 = (__int64 *)(v10 + 16LL * v12);
          v14 = (_QWORD *)*v13;
          if ( (_QWORD *)*v13 == v5 )
            goto LABEL_10;
          v108 = v4;
        }
LABEL_4:
        v7 = v5[4];
        if ( v7 != v8 + 48 )
          goto LABEL_5;
LABEL_14:
        v5 = 0;
        if ( !*(_QWORD *)(v3 + 168) )
        {
LABEL_15:
          v109 = *(_QWORD **)(v3 + 160);
          if ( v109 != v5 )
          {
            v17 = *(_QWORD **)(v3 + 160);
            while ( 1 )
            {
              v19 = v17[5];
              if ( *(_QWORD *)v3 == v19 )
              {
                v20 = *(_DWORD *)(v3 + 104);
                v21 = *(_QWORD *)(v3 + 88);
                if ( v20 )
                {
                  v22 = v20 - 1;
                  v23 = v22 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
                  v24 = (__int64 *)(v21 + 16LL * v23);
                  v25 = (_QWORD *)*v24;
                  if ( v17 == (_QWORD *)*v24 )
                  {
LABEL_23:
                    v26 = v24[1];
                    if ( v26 )
                    {
                      if ( *(_DWORD *)(v26 + 136) == *(_DWORD *)(v3 + 204) )
                      {
                        v114 = v26;
                        if ( *(_QWORD *)(v26 + 16) == v26 && *(_DWORD *)(v26 + 144) != -1 )
                        {
                          v27 = v26;
                          v28 = 0;
                          while ( 1 )
                          {
                            v29 = *(_DWORD *)(v27 + 148);
                            if ( v29 == -1 )
                              break;
                            v27 = *(_QWORD *)(v27 + 24);
                            v28 += v29;
                            if ( !v27 )
                            {
                              if ( v28 )
                                break;
                              if ( *(_BYTE *)(v26 + 152) )
                                break;
                              v31 = sub_2B08790((__int64)&v115, (__int64)&v114);
                              if ( !v30 )
                                break;
                              sub_2B08E40((__int64)&v115, (__int64)v31, v30, (__int64)&v114);
                              v19 = v17[5];
                              v18 = v17[4];
                              if ( v18 )
                                goto LABEL_18;
                              goto LABEL_34;
                            }
                          }
                        }
                      }
                    }
                  }
                  else
                  {
                    v106 = 1;
                    while ( v25 != (_QWORD *)-4096LL )
                    {
                      v107 = v106 + 1;
                      v23 = v22 & (v106 + v23);
                      v24 = (__int64 *)(v21 + 16LL * v23);
                      v25 = (_QWORD *)*v24;
                      if ( (_QWORD *)*v24 == v17 )
                        goto LABEL_23;
                      v106 = v107;
                    }
                  }
                }
              }
              v18 = v17[4];
              if ( !v18 )
                goto LABEL_34;
LABEL_18:
              if ( v18 == v19 + 48 )
              {
LABEL_34:
                v17 = 0;
                if ( !*(_QWORD *)(v3 + 168) )
                {
LABEL_35:
                  v109 = v17;
                  v32 = v120;
                  goto LABEL_36;
                }
              }
              else
              {
                v17 = (_QWORD *)(v18 - 24);
                if ( *(_QWORD **)(v3 + 168) == v17 )
                  goto LABEL_35;
              }
            }
          }
          v32 = v120;
LABEL_36:
          v33 = &v114;
          if ( !v32 )
            goto LABEL_87;
          v112 = v3;
LABEL_38:
          v34 = v118[4];
          v35 = sub_220F330((int *)v118, &v116);
          j_j___libc_free_0((unsigned __int64)v35);
          --v120;
          if ( v34 )
          {
            v36 = v109;
            v37 = (__int64 *)v34;
            do
            {
              v38 = v36;
              v36 = (_QWORD *)*v37;
              if ( v38 != (_QWORD *)sub_B46B10(*v37, 0) )
              {
                if ( *(_QWORD **)(v38[5] + 56LL) == v38 + 3 || (v38[3] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  v39 = 0;
                else
                  v39 = (v38[3] & 0xFFFFFFFFFFFFFFF8LL) - 24;
                sub_B44530(v36, v39);
              }
              v37 = (__int64 *)v37[3];
            }
            while ( v37 );
            v109 = v36;
          }
          *(_BYTE *)(v34 + 152) = 1;
          while ( 2 )
          {
            v40 = *(_QWORD *)(v34 + 8);
            v41 = *(_QWORD *)v34;
            if ( v40 )
            {
              v42 = *(_QWORD **)v40;
              v43 = *(unsigned int *)(v40 + 8);
              v113 = *(_QWORD *)v34;
              v44 = 0;
              v45 = sub_2B0C7B0(v42, (__int64)&v42[v43], &v113);
              v48 = (__int64)v45 - v47;
              v49 = *(unsigned int *)(v46 + 248);
              if ( !*(_DWORD *)(v46 + 248) )
                goto LABEL_67;
              v50 = (__int64)v33;
              v51 = 8LL * (int)(v48 >> 3);
              v52 = v46;
              v53 = v34;
              v54 = v49;
              while ( 1 )
              {
                v55 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v52 + 240) + 80LL * (unsigned int)v44) + v51);
                if ( *(_BYTE *)v55 <= 0x1Cu )
                  goto LABEL_51;
                if ( *(_QWORD *)v112 != *(_QWORD *)(v55 + 40) )
                  goto LABEL_51;
                v56 = *(_DWORD *)(v112 + 104);
                v57 = *(_QWORD *)(v112 + 88);
                if ( !v56 )
                  goto LABEL_51;
                v58 = v56 - 1;
                v59 = v58 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                v60 = (__int64 *)(v57 + 16LL * v59);
                v61 = *v60;
                if ( v55 != *v60 )
                {
                  v102 = 1;
                  while ( v61 != -4096 )
                  {
                    v103 = v102 + 1;
                    v59 = v58 & (v102 + v59);
                    v60 = (__int64 *)(v57 + 16LL * v59);
                    v61 = *v60;
                    if ( v55 == *v60 )
                      goto LABEL_56;
                    v102 = v103;
                  }
                  goto LABEL_51;
                }
LABEL_56:
                v62 = v60[1];
                if ( !v62 || *(_DWORD *)(v62 + 136) != *(_DWORD *)(v112 + 204) || *(_DWORD *)(v62 + 144) == -1 )
                  goto LABEL_51;
                v63 = *(_QWORD *)(v62 + 16);
                --*(_DWORD *)(v62 + 148);
                if ( v63 )
                {
                  v64 = v63;
                  v65 = 0;
                  while ( 1 )
                  {
                    v66 = *(_DWORD *)(v64 + 148);
                    if ( v66 == -1 )
                      goto LABEL_51;
                    v64 = *(_QWORD *)(v64 + 24);
                    v65 += v66;
                    if ( !v64 )
                    {
                      if ( v65 )
                        goto LABEL_51;
                      break;
                    }
                  }
                }
                v114 = v63;
                v110 = v53;
                v111 = v50;
                v67 = sub_2B08790((__int64)&v115, v50);
                v50 = v111;
                v53 = v110;
                if ( v68 )
                {
                  ++v44;
                  sub_2B08E40((__int64)&v115, (__int64)v67, v68, v111);
                  v50 = v111;
                  v53 = v110;
                  if ( v54 == v44 )
                  {
LABEL_66:
                    v34 = v53;
                    v33 = (__int64 *)v50;
                    goto LABEL_67;
                  }
                }
                else
                {
LABEL_51:
                  if ( v54 == ++v44 )
                    goto LABEL_66;
                }
              }
            }
            v86 = 4LL * (*(_DWORD *)(v41 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v41 + 7) & 0x40) != 0 )
            {
              v87 = *(_QWORD **)(v41 - 8);
              v41 = (__int64)&v87[v86];
            }
            else
            {
              v87 = (_QWORD *)(v41 - v86 * 8);
            }
            for ( ; (_QWORD *)v41 != v87; v87 += 4 )
            {
              v88 = (_BYTE *)*v87;
              if ( *(_BYTE *)*v87 > 0x1Cu && *(_QWORD *)v112 == *((_QWORD *)v88 + 5) )
              {
                v89 = *(_DWORD *)(v112 + 104);
                v90 = *(_QWORD *)(v112 + 88);
                if ( v89 )
                {
                  v91 = v89 - 1;
                  v92 = v91 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
                  v93 = (_QWORD *)(v90 + 16LL * v92);
                  v94 = (_BYTE *)*v93;
                  if ( v88 == (_BYTE *)*v93 )
                  {
LABEL_103:
                    v95 = v93[1];
                    if ( v95 && *(_DWORD *)(v95 + 136) == *(_DWORD *)(v112 + 204) && *(_DWORD *)(v95 + 144) != -1 )
                    {
                      v96 = *(_QWORD *)(v95 + 16);
                      --*(_DWORD *)(v95 + 148);
                      if ( v96 )
                      {
                        v97 = v96;
                        v98 = 0;
                        while ( 1 )
                        {
                          v99 = *(_DWORD *)(v97 + 148);
                          if ( v99 == -1 )
                            break;
                          v97 = *(_QWORD *)(v97 + 24);
                          v98 += v99;
                          if ( !v97 )
                          {
                            if ( v98 )
                              break;
                            goto LABEL_111;
                          }
                        }
                      }
                      else
                      {
LABEL_111:
                        v114 = v96;
                        v101 = sub_2B08790((__int64)&v115, (__int64)v33);
                        if ( v100 )
                          sub_2B08E40((__int64)&v115, (__int64)v101, v100, (__int64)v33);
                      }
                    }
                  }
                  else
                  {
                    v104 = 1;
                    while ( v94 != (_BYTE *)-4096LL )
                    {
                      v105 = v104 + 1;
                      v92 = v91 & (v104 + v92);
                      v93 = (_QWORD *)(v90 + 16LL * v92);
                      v94 = (_BYTE *)*v93;
                      if ( v88 == (_BYTE *)*v93 )
                        goto LABEL_103;
                      v104 = v105;
                    }
                  }
                }
              }
            }
LABEL_67:
            v69 = *(__int64 **)(v34 + 40);
            v70 = &v69[*(unsigned int *)(v34 + 48)];
            if ( v69 == v70 )
            {
LABEL_78:
              v78 = *(_QWORD *)(v34 + 88);
              v79 = v78 + 8LL * *(unsigned int *)(v34 + 96);
              if ( v78 != v79 )
              {
                while ( 1 )
                {
                  v80 = *(_QWORD *)(*(_QWORD *)v78 + 16LL);
                  --*(_DWORD *)(*(_QWORD *)v78 + 148LL);
                  if ( v80 )
                    break;
LABEL_91:
                  v114 = v80;
                  v85 = sub_2B08790((__int64)&v115, (__int64)v33);
                  if ( v84 )
                  {
                    v78 += 8;
                    sub_2B08E40((__int64)&v115, (__int64)v85, v84, (__int64)v33);
                    if ( v79 == v78 )
                      goto LABEL_84;
                  }
                  else
                  {
LABEL_83:
                    v78 += 8;
                    if ( v79 == v78 )
                      goto LABEL_84;
                  }
                }
                v81 = v80;
                v82 = 0;
                while ( 1 )
                {
                  v83 = *(_DWORD *)(v81 + 148);
                  if ( v83 == -1 )
                    goto LABEL_83;
                  v81 = *(_QWORD *)(v81 + 24);
                  v82 += v83;
                  if ( !v81 )
                  {
                    if ( v82 )
                      goto LABEL_83;
                    goto LABEL_91;
                  }
                }
              }
LABEL_84:
              v34 = *(_QWORD *)(v34 + 24);
              if ( v34 )
                continue;
              if ( v120 )
                goto LABEL_38;
              v3 = v112;
LABEL_87:
              *(_QWORD *)(v3 + 160) = 0;
              sub_2B106E0(v117);
              return;
            }
            break;
          }
          while ( 1 )
          {
LABEL_70:
            v71 = *v69;
            if ( *(_DWORD *)(*v69 + 144) == -1 )
              goto LABEL_69;
            v72 = *(_QWORD *)(v71 + 16);
            --*(_DWORD *)(v71 + 148);
            if ( v72 )
              break;
LABEL_76:
            v114 = v72;
            v77 = sub_2B08790((__int64)&v115, (__int64)v33);
            if ( !v76 )
              goto LABEL_69;
            ++v69;
            sub_2B08E40((__int64)&v115, (__int64)v77, v76, (__int64)v33);
            if ( v70 == v69 )
              goto LABEL_78;
          }
          v73 = v72;
          v74 = 0;
          while ( 1 )
          {
            v75 = *(_DWORD *)(v73 + 148);
            if ( v75 == -1 )
              break;
            v73 = *(_QWORD *)(v73 + 24);
            v74 += v75;
            if ( !v73 )
            {
              if ( v74 )
                break;
              goto LABEL_76;
            }
          }
LABEL_69:
          if ( v70 == ++v69 )
            goto LABEL_78;
          goto LABEL_70;
        }
      }
    }
  }
}
