// Function: sub_30BF1A0
// Address: 0x30bf1a0
//
__int64 *__fastcall sub_30BF1A0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 *v2; // rcx
  __int64 *result; // rax
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 (__fastcall *v10)(__int64, __int64); // rbx
  char v11; // r12
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 (__fastcall *v15)(__int64, __int64, __int64); // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned int v18; // esi
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // rdi
  int v22; // r10d
  _QWORD *v23; // r11
  unsigned int v24; // r12d
  unsigned int v25; // ecx
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 (__fastcall *v28)(__int64, __int64, __int64); // rax
  __int64 v29; // rax
  __int64 v30; // rbx
  unsigned int v31; // esi
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rdi
  int v35; // r13d
  _QWORD *v36; // r10
  unsigned int v37; // r12d
  unsigned int v38; // ecx
  _QWORD *v39; // rdx
  __int64 v40; // rax
  void (*v41)(void); // rax
  int v42; // eax
  int v43; // edx
  __int64 v44; // rax
  int v45; // eax
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // r12
  __int64 (__fastcall *v50)(__int64, __int64, __int64); // rax
  __int64 v51; // rax
  void (__fastcall *v52)(unsigned __int64); // rax
  unsigned int i; // ebx
  unsigned int v54; // esi
  __int64 (__fastcall *v55)(__int64, __int64, __int64); // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 (__fastcall *v59)(__int64, __int64, __int64); // rax
  __int64 v60; // rax
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // rdi
  unsigned int v64; // edx
  __int64 v65; // rsi
  int v66; // r11d
  int v67; // r8d
  __int64 v68; // rsi
  unsigned int v69; // eax
  __int64 v70; // rcx
  int v71; // r10d
  int v72; // edx
  int v73; // edx
  __int64 v74; // rsi
  unsigned int v75; // r12d
  __int64 v76; // rcx
  int v77; // edi
  int v78; // edi
  __int64 v79; // rcx
  unsigned int v80; // r12d
  __int64 v81; // rax
  __int64 *v82; // [rsp+30h] [rbp-E0h]
  __int64 v83; // [rsp+38h] [rbp-D8h]
  __int64 *v84; // [rsp+40h] [rbp-D0h]
  __int64 *v85; // [rsp+50h] [rbp-C0h]
  __int64 *v86; // [rsp+58h] [rbp-B8h]
  __int64 *v87; // [rsp+60h] [rbp-B0h]
  __int64 v88; // [rsp+68h] [rbp-A8h]
  __int64 v89; // [rsp+68h] [rbp-A8h]
  __int64 (__fastcall *v90)(__int64, __int64); // [rsp+68h] [rbp-A8h]
  __int64 v91; // [rsp+68h] [rbp-A8h]
  __int64 v92; // [rsp+68h] [rbp-A8h]
  char v93; // [rsp+70h] [rbp-A0h]
  __int64 v94; // [rsp+70h] [rbp-A0h]
  __int64 (__fastcall *v95)(__int64, __int64); // [rsp+78h] [rbp-98h]
  char v96; // [rsp+87h] [rbp-89h] BYREF
  unsigned __int64 v97; // [rsp+88h] [rbp-88h] BYREF
  __int64 v98[2]; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v99; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v100; // [rsp+A8h] [rbp-68h]
  _BYTE v101[16]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 (__fastcall *v102)(__int64, __int64); // [rsp+C0h] [rbp-50h] BYREF
  __int64 v103; // [rsp+C8h] [rbp-48h]
  _BYTE v104[64]; // [rsp+D0h] [rbp-40h] BYREF

  v1 = a1[1];
  v2 = *(__int64 **)(v1 + 96);
  result = &v2[*(unsigned int *)(v1 + 104)];
  v86 = v2;
  v82 = result;
  if ( v2 == result )
    return result;
  while ( 2 )
  {
    v100 = 0x200000000LL;
    v99 = (__int64 *)v101;
    v6 = *v86;
    v102 = sub_30B9670;
    v103 = (__int64)&v96;
    sub_30B0A30(v6, (__int64)&v102, (__int64)&v99);
    if ( !(_DWORD)v100 )
    {
      v5 = (unsigned __int64)v99;
      if ( v99 == (__int64 *)v101 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v87 = v86;
    while ( 2 )
    {
      if ( *v87 + 8 == *v86 + 8 )
        goto LABEL_10;
      v103 = 0x200000000LL;
      v102 = (__int64 (__fastcall *)(__int64, __int64))v104;
      v8 = *v87;
      v98[0] = (__int64)sub_30B9670;
      v98[1] = (__int64)&v96;
      sub_30B0A30(v8, (__int64)v98, (__int64)&v102);
      v9 = (unsigned int)v103;
      if ( !(_DWORD)v103 )
      {
        v7 = (unsigned __int64)v102;
        if ( (char *)v102 != v104 )
          goto LABEL_9;
        goto LABEL_10;
      }
      v10 = v102;
      v84 = &v99[(unsigned int)v100];
      if ( v84 == v99 )
        goto LABEL_34;
      v85 = v99;
      v11 = 0;
      v93 = 0;
      while ( 1 )
      {
        v95 = (__int64 (__fastcall *)(__int64, __int64))((char *)v10 + 8 * v9);
        v12 = *v85;
        if ( v95 != v10 )
          break;
LABEL_78:
        if ( v11 && v93 )
          goto LABEL_34;
        if ( v84 == ++v85 )
          goto LABEL_34;
        v9 = (unsigned int)v103;
      }
      while ( 1 )
      {
        sub_2297CA0((__int64 *)&v97, a1[2], v12, *(_BYTE **)v10);
        if ( !v97 )
          goto LABEL_76;
        if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64))(*(_QWORD *)v97 + 24LL))(v97) )
        {
          v13 = *v87;
          v14 = *v86;
          if ( v11 )
            goto LABEL_24;
          v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
          if ( v15 != sub_30B2FE0 )
          {
            v15((__int64)a1, *v86, *v87);
            goto LABEL_24;
          }
          v16 = sub_22077B0(0x10u);
          v17 = v16;
          if ( v16 )
          {
            *(_QWORD *)v16 = v13;
            *(_DWORD *)(v16 + 8) = 2;
          }
          v18 = *(_DWORD *)(v14 + 32);
          v19 = v14 + 8;
          if ( v18 )
          {
            v20 = v18 - 1;
            v21 = *(_QWORD *)(v14 + 16);
            v22 = 1;
            v23 = 0;
            v24 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
            v25 = v20 & v24;
            v26 = (_QWORD *)(v21 + 8LL * ((unsigned int)v20 & v24));
            v27 = *v26;
            if ( v17 == *v26 )
              goto LABEL_24;
            while ( v27 != -4096 )
            {
              if ( v27 != -8192 || v23 )
                v26 = v23;
              v25 = v20 & (v22 + v25);
              v27 = *(_QWORD *)(v21 + 8LL * v25);
              if ( v17 == v27 )
                goto LABEL_24;
              ++v22;
              v23 = v26;
              v26 = (_QWORD *)(v21 + 8LL * v25);
            }
            v42 = *(_DWORD *)(v14 + 24);
            if ( !v23 )
              v23 = v26;
            ++*(_QWORD *)(v14 + 8);
            v43 = v42 + 1;
            if ( 4 * (v42 + 1) < 3 * v18 )
            {
              if ( v18 - *(_DWORD *)(v14 + 28) - v43 > v18 >> 3 )
              {
LABEL_45:
                *(_DWORD *)(v14 + 24) = v43;
                if ( *v23 != -4096 )
                  --*(_DWORD *)(v14 + 28);
                *v23 = v17;
                v44 = *(unsigned int *)(v14 + 48);
                if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(v14 + 52) )
                {
                  sub_C8D5F0(v14 + 40, (const void *)(v14 + 56), v44 + 1, 8u, v20, v19);
                  v44 = *(unsigned int *)(v14 + 48);
                }
                *(_QWORD *)(*(_QWORD *)(v14 + 40) + 8 * v44) = v17;
                ++*(_DWORD *)(v14 + 48);
LABEL_24:
                if ( v93 )
                  goto LABEL_30;
                v28 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
                if ( v28 != sub_30B2FE0 )
                  goto LABEL_101;
                v29 = sub_22077B0(0x10u);
                v30 = v29;
                if ( v29 )
                {
                  *(_QWORD *)v29 = v14;
                  *(_DWORD *)(v29 + 8) = 2;
                }
                v31 = *(_DWORD *)(v13 + 32);
                v32 = v13 + 8;
                if ( v31 )
                {
                  v33 = v31 - 1;
                  v34 = *(_QWORD *)(v13 + 16);
                  v35 = 1;
                  v36 = 0;
                  v37 = ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4);
                  v38 = v33 & v37;
                  v39 = (_QWORD *)(v34 + 8LL * ((unsigned int)v33 & v37));
                  v40 = *v39;
                  if ( *v39 == v30 )
                    goto LABEL_30;
                  while ( v40 != -4096 )
                  {
                    if ( v40 != -8192 || v36 )
                      v39 = v36;
                    v38 = v33 & (v35 + v38);
                    v40 = *(_QWORD *)(v34 + 8LL * v38);
                    if ( v30 == v40 )
                      goto LABEL_30;
                    ++v35;
                    v36 = v39;
                    v39 = (_QWORD *)(v34 + 8LL * v38);
                  }
                  v45 = *(_DWORD *)(v13 + 24);
                  if ( !v36 )
                    v36 = v39;
                  ++*(_QWORD *)(v13 + 8);
                  v46 = v45 + 1;
                  if ( 4 * v46 < 3 * v31 )
                  {
                    if ( v31 - *(_DWORD *)(v13 + 28) - v46 > v31 >> 3 )
                    {
LABEL_59:
                      *(_DWORD *)(v13 + 24) = v46;
                      if ( *v36 != -4096 )
                        --*(_DWORD *)(v13 + 28);
                      *v36 = v30;
                      v47 = *(unsigned int *)(v13 + 48);
                      if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 52) )
                      {
                        sub_C8D5F0(v13 + 40, (const void *)(v13 + 56), v47 + 1, 8u, v33, v32);
                        v47 = *(unsigned int *)(v13 + 48);
                      }
                      *(_QWORD *)(*(_QWORD *)(v13 + 40) + 8 * v47) = v30;
                      ++*(_DWORD *)(v13 + 48);
                      goto LABEL_30;
                    }
                    sub_30B28F0(v13 + 8, v31);
                    v72 = *(_DWORD *)(v13 + 32);
                    if ( v72 )
                    {
                      v73 = v72 - 1;
                      v74 = *(_QWORD *)(v13 + 16);
                      v32 = 1;
                      v33 = 0;
                      v75 = v73 & v37;
                      v36 = (_QWORD *)(v74 + 8LL * v75);
                      v76 = *v36;
                      v46 = *(_DWORD *)(v13 + 24) + 1;
                      if ( v30 != *v36 )
                      {
                        while ( v76 != -4096 )
                        {
                          if ( v76 == -8192 && !v33 )
                            v33 = (__int64)v36;
                          v75 = v73 & (v32 + v75);
                          v36 = (_QWORD *)(v74 + 8LL * v75);
                          v76 = *v36;
                          if ( v30 == *v36 )
                            goto LABEL_59;
                          v32 = (unsigned int)(v32 + 1);
                        }
                        if ( v33 )
                          v36 = (_QWORD *)v33;
                      }
                      goto LABEL_59;
                    }
LABEL_177:
                    ++*(_DWORD *)(v13 + 24);
                    BUG();
                  }
                }
                else
                {
                  ++*(_QWORD *)(v13 + 8);
                }
                sub_30B28F0(v13 + 8, 2 * v31);
                v61 = *(_DWORD *)(v13 + 32);
                if ( v61 )
                {
                  v62 = v61 - 1;
                  v63 = *(_QWORD *)(v13 + 16);
                  v64 = v62 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                  v36 = (_QWORD *)(v63 + 8LL * v64);
                  v65 = *v36;
                  v46 = *(_DWORD *)(v13 + 24) + 1;
                  if ( *v36 != v30 )
                  {
                    v66 = 1;
                    v32 = 0;
                    while ( v65 != -4096 )
                    {
                      if ( !v32 && v65 == -8192 )
                        v32 = (__int64)v36;
                      v33 = (unsigned int)(v66 + 1);
                      v64 = v62 & (v66 + v64);
                      v36 = (_QWORD *)(v63 + 8LL * v64);
                      v65 = *v36;
                      if ( v30 == *v36 )
                        goto LABEL_59;
                      ++v66;
                    }
                    if ( v32 )
                      v36 = (_QWORD *)v32;
                  }
                  goto LABEL_59;
                }
                goto LABEL_177;
              }
              sub_30B28F0(v14 + 8, v18);
              v77 = *(_DWORD *)(v14 + 32);
              if ( v77 )
              {
                v78 = v77 - 1;
                v79 = *(_QWORD *)(v14 + 16);
                v20 = 0;
                v80 = v78 & v24;
                v19 = 1;
                v43 = *(_DWORD *)(v14 + 24) + 1;
                v23 = (_QWORD *)(v79 + 8LL * v80);
                v81 = *v23;
                if ( v17 != *v23 )
                {
                  while ( v81 != -4096 )
                  {
                    if ( !v20 && v81 == -8192 )
                      v20 = (__int64)v23;
                    v80 = v78 & (v19 + v80);
                    v23 = (_QWORD *)(v79 + 8LL * v80);
                    v81 = *v23;
                    if ( v17 == *v23 )
                      goto LABEL_45;
                    v19 = (unsigned int)(v19 + 1);
                  }
                  if ( v20 )
                    v23 = (_QWORD *)v20;
                }
                goto LABEL_45;
              }
LABEL_178:
              ++*(_DWORD *)(v14 + 24);
              BUG();
            }
          }
          else
          {
            ++*(_QWORD *)(v14 + 8);
          }
          sub_30B28F0(v14 + 8, 2 * v18);
          v67 = *(_DWORD *)(v14 + 32);
          if ( v67 )
          {
            v20 = (unsigned int)(v67 - 1);
            v68 = *(_QWORD *)(v14 + 16);
            v69 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v23 = (_QWORD *)(v68 + 8LL * v69);
            v70 = *v23;
            v43 = *(_DWORD *)(v14 + 24) + 1;
            if ( v17 != *v23 )
            {
              v71 = 1;
              v19 = 0;
              while ( v70 != -4096 )
              {
                if ( v70 == -8192 && !v19 )
                  v19 = (__int64)v23;
                v69 = v20 & (v71 + v69);
                v23 = (_QWORD *)(v68 + 8LL * v69);
                v70 = *v23;
                if ( v17 == *v23 )
                  goto LABEL_45;
                ++v71;
              }
              if ( v19 )
                v23 = (_QWORD *)v19;
            }
            goto LABEL_45;
          }
          goto LABEL_178;
        }
        v88 = v97;
        v48 = v97;
        if ( ((unsigned __int8)sub_228CC50(v97) || (unsigned __int8)sub_228CC90(v48)
                                                || (unsigned __int8)sub_228CCD0(v88))
          && !(*(unsigned __int8 (__fastcall **)(unsigned __int64))(*(_QWORD *)v97 + 16LL))(v97) )
        {
          break;
        }
        if ( !v11 )
        {
          v49 = *v86;
          v50 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
          if ( v50 == sub_30B2FE0 )
          {
            v89 = *v87;
            v51 = sub_22077B0(0x10u);
            if ( v51 )
            {
              *(_DWORD *)(v51 + 8) = 2;
              *(_QWORD *)v51 = v89;
            }
            v98[0] = v51;
            sub_30B2AC0(v49 + 8, v98);
          }
          else
          {
LABEL_103:
            v50((__int64)a1, *v86, *v87);
          }
        }
LABEL_71:
        if ( v93 )
          goto LABEL_30;
        v11 = 1;
LABEL_73:
        if ( v97 )
        {
          v52 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v97 + 8LL);
          if ( v52 == sub_228A6E0 )
          {
            j_j___libc_free_0(v97);
            goto LABEL_76;
          }
          ((void (*)(void))v52)();
          v10 = (__int64 (__fastcall *)(__int64, __int64))((char *)v10 + 8);
          if ( v95 == v10 )
          {
LABEL_77:
            v10 = v102;
            goto LABEL_78;
          }
        }
        else
        {
LABEL_76:
          v10 = (__int64 (__fastcall *)(__int64, __int64))((char *)v10 + 8);
          if ( v95 == v10 )
            goto LABEL_77;
        }
      }
      v90 = v10;
      for ( i = 1; ; ++i )
      {
        if ( (*(unsigned int (__fastcall **)(unsigned __int64))(*(_QWORD *)v97 + 40LL))(v97) < i )
        {
          v10 = v90;
LABEL_105:
          if ( !v11 )
          {
            v50 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
            if ( v50 != sub_30B2FE0 )
              goto LABEL_103;
            v83 = *v86;
            v91 = *v87;
            v58 = sub_22077B0(0x10u);
            if ( v58 )
            {
              *(_DWORD *)(v58 + 8) = 2;
              *(_QWORD *)v58 = v91;
            }
            v98[0] = v58;
            sub_30B2AC0(v83 + 8, v98);
          }
          goto LABEL_71;
        }
        if ( (*(unsigned int (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v97 + 48LL))(v97, i) != 2 )
          break;
      }
      v54 = i;
      v10 = v90;
      if ( (*(unsigned int (__fastcall **)(unsigned __int64))(*(_QWORD *)v97 + 48LL))(v97) == 4 )
      {
        if ( !v93 )
        {
          v59 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
          if ( v59 == sub_30B2FE0 )
          {
            v94 = *v87;
            v92 = *v86;
            v60 = sub_22077B0(0x10u);
            if ( v60 )
            {
              *(_DWORD *)(v60 + 8) = 2;
              *(_QWORD *)v60 = v92;
            }
            v98[0] = v60;
            sub_30B2AC0(v94 + 8, v98);
          }
          else
          {
            v59((__int64)a1, *v87, *v86);
          }
        }
        if ( v11 )
          goto LABEL_30;
        v93 = 1;
        goto LABEL_73;
      }
      if ( (*(unsigned int (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v97 + 48LL))(v97, v54) == 1 )
        goto LABEL_105;
      v13 = *v87;
      v14 = *v86;
      if ( !v11 )
      {
        v55 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
        if ( v55 == sub_30B2FE0 )
        {
          v56 = sub_22077B0(0x10u);
          if ( v56 )
          {
            *(_QWORD *)v56 = v13;
            *(_DWORD *)(v56 + 8) = 2;
          }
          v98[0] = v56;
          sub_30B2AC0(v14 + 8, v98);
        }
        else
        {
          v55((__int64)a1, *v86, *v87);
        }
      }
      if ( !v93 )
      {
        v28 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 48LL);
        if ( v28 == sub_30B2FE0 )
        {
          v57 = sub_22077B0(0x10u);
          if ( v57 )
          {
            *(_QWORD *)v57 = v14;
            *(_DWORD *)(v57 + 8) = 2;
          }
          v98[0] = v57;
          sub_30B2AC0(v13 + 8, v98);
          goto LABEL_30;
        }
LABEL_101:
        v28((__int64)a1, v13, v14);
      }
LABEL_30:
      if ( v97 )
      {
        v41 = *(void (**)(void))(*(_QWORD *)v97 + 8LL);
        if ( (char *)v41 == (char *)sub_228A6E0 )
          j_j___libc_free_0(v97);
        else
          v41();
      }
      v10 = v102;
LABEL_34:
      if ( (char *)v10 != v104 )
      {
        v7 = (unsigned __int64)v10;
LABEL_9:
        _libc_free(v7);
      }
LABEL_10:
      if ( v82 != ++v87 )
        continue;
      break;
    }
    v5 = (unsigned __int64)v99;
    if ( v99 != (__int64 *)v101 )
LABEL_4:
      _libc_free(v5);
LABEL_5:
    result = ++v86;
    if ( v82 != v86 )
      continue;
    return result;
  }
}
