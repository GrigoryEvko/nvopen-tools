// Function: sub_D218F0
// Address: 0xd218f0
//
__int64 __fastcall sub_D218F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r11d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // r15
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rdx
  bool v18; // zf
  unsigned int v19; // eax
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *j; // rdx
  int v25; // r14d
  unsigned int v26; // eax
  _QWORD *v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // r13
  _QWORD *v30; // rcx
  unsigned __int64 v31; // r13
  int v32; // eax
  __int64 v33; // rdx
  _QWORD *v34; // rax
  _QWORD *m; // rdx
  _QWORD *v36; // r14
  _QWORD *v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 v44; // r10
  unsigned int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // rdi
  int v48; // ebx
  _QWORD *v49; // rax
  unsigned int v50; // ecx
  unsigned int v51; // eax
  _QWORD *v52; // rdi
  int v53; // ebx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // rdx
  _QWORD *ii; // rdx
  unsigned int v59; // edx
  int v60; // ebx
  unsigned int v61; // r14d
  unsigned int v62; // eax
  _QWORD *v63; // rdi
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rdi
  _QWORD *v66; // rax
  __int64 v67; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rdi
  _QWORD *v71; // rax
  __int64 v72; // rdx
  _QWORD *n; // rdx
  _QWORD *v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // [rsp+8h] [rbp-58h]
  _QWORD *v77; // [rsp+10h] [rbp-50h]
  __int64 v78; // [rsp+18h] [rbp-48h]
  _QWORD *v80; // [rsp+28h] [rbp-38h]
  _QWORD *v81; // [rsp+28h] [rbp-38h]
  _QWORD *v82; // [rsp+28h] [rbp-38h]
  _QWORD *v83; // [rsp+28h] [rbp-38h]
  unsigned __int64 v84; // [rsp+28h] [rbp-38h]

  v6 = a3;
  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  if ( (_DWORD)v7 )
  {
    v9 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((v6 >> 9) ^ (v6 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F86B78 >> 9) ^ ((unsigned int)&unk_4F86B78 >> 4)) << 32))) >> 31)
             ^ (484763065 * ((v6 >> 9) ^ (v6 >> 4)))); ; i = (v7 - 1) & v12 )
    {
      v11 = v8 + 24LL * i;
      if ( *(_UNKNOWN **)v11 == &unk_4F86B78 && a3 == *(_QWORD *)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_52;
      v12 = v9 + i;
      ++v9;
    }
    if ( v11 != v8 + 24 * v7 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
      if ( v13 )
      {
        v14 = (__int64)&unk_4F86A90;
        v78 = v13 + 8;
        v15 = sub_BC0510(a4, &unk_4F86A90, a3);
        ++*(_QWORD *)(v13 + 48);
        v77 = (_QWORD *)(v15 + 8);
        if ( !*(_BYTE *)(v13 + 76) )
        {
          v16 = 4 * (*(_DWORD *)(v13 + 68) - *(_DWORD *)(v13 + 72));
          v17 = *(unsigned int *)(v13 + 64);
          if ( v16 < 0x20 )
            v16 = 32;
          if ( v16 < (unsigned int)v17 )
          {
            sub_C8C990(v13 + 48, (__int64)&unk_4F86A90);
LABEL_15:
            ++*(_QWORD *)(v13 + 152);
            v18 = *(_BYTE *)(v13 + 180) == 0;
            *(_BYTE *)(v13 + 144) = 0;
            if ( v18 )
            {
              v19 = 4 * (*(_DWORD *)(v13 + 172) - *(_DWORD *)(v13 + 176));
              v20 = *(unsigned int *)(v13 + 168);
              if ( v19 < 0x20 )
                v19 = 32;
              if ( v19 < (unsigned int)v20 )
              {
                sub_C8C990(v13 + 152, v14);
LABEL_21:
                v21 = *(_DWORD *)(v13 + 264);
                ++*(_QWORD *)(v13 + 248);
                if ( !v21 )
                {
                  if ( !*(_DWORD *)(v13 + 268) )
                    goto LABEL_27;
                  v22 = *(unsigned int *)(v13 + 272);
                  if ( (unsigned int)v22 > 0x40 )
                  {
                    sub_C7D6A0(*(_QWORD *)(v13 + 256), 16LL * (unsigned int)v22, 8);
                    *(_QWORD *)(v13 + 256) = 0;
                    *(_QWORD *)(v13 + 264) = 0;
                    *(_DWORD *)(v13 + 272) = 0;
                    goto LABEL_27;
                  }
                  goto LABEL_24;
                }
                v50 = 4 * v21;
                v22 = *(unsigned int *)(v13 + 272);
                if ( (unsigned int)(4 * v21) < 0x40 )
                  v50 = 64;
                if ( v50 >= (unsigned int)v22 )
                {
LABEL_24:
                  v23 = *(_QWORD **)(v13 + 256);
                  for ( j = &v23[2 * v22]; j != v23; v23 += 2 )
                    *v23 = -4096;
                  *(_QWORD *)(v13 + 264) = 0;
                  goto LABEL_27;
                }
                v51 = v21 - 1;
                if ( v51 )
                {
                  _BitScanReverse(&v51, v51);
                  v52 = *(_QWORD **)(v13 + 256);
                  v53 = 1 << (33 - (v51 ^ 0x1F));
                  if ( v53 < 64 )
                    v53 = 64;
                  if ( v53 == (_DWORD)v22 )
                  {
                    *(_QWORD *)(v13 + 264) = 0;
                    v75 = &v52[2 * (unsigned int)v53];
                    do
                    {
                      if ( v52 )
                        *v52 = -4096;
                      v52 += 2;
                    }
                    while ( v75 != v52 );
LABEL_27:
                    v25 = *(_DWORD *)(v13 + 296);
                    ++*(_QWORD *)(v13 + 280);
                    if ( !v25 && !*(_DWORD *)(v13 + 300) )
                      goto LABEL_40;
                    v26 = 4 * v25;
                    v27 = *(_QWORD **)(v13 + 288);
                    v28 = *(unsigned int *)(v13 + 304);
                    v29 = 16 * v28;
                    if ( (unsigned int)(4 * v25) < 0x40 )
                      v26 = 64;
                    v30 = &v27[(unsigned __int64)v29 / 8];
                    if ( (unsigned int)v28 <= v26 )
                    {
                      while ( v27 != v30 )
                      {
                        if ( *v27 != -4096 )
                        {
                          if ( *v27 != -8192 )
                          {
                            v31 = v27[1] & 0xFFFFFFFFFFFFFFF8LL;
                            if ( v31 )
                            {
                              if ( (*(_BYTE *)(v31 + 8) & 1) == 0 )
                              {
                                v82 = v30;
                                sub_C7D6A0(*(_QWORD *)(v31 + 16), 16LL * *(unsigned int *)(v31 + 24), 8);
                                v30 = v82;
                              }
                              v83 = v30;
                              j_j___libc_free_0(v31, 272);
                              v30 = v83;
                            }
                          }
                          *v27 = -4096;
                        }
                        v27 += 2;
                      }
                      goto LABEL_39;
                    }
                    while ( 1 )
                    {
                      while ( *v27 == -8192 )
                      {
LABEL_57:
                        v27 += 2;
                        if ( v27 == v30 )
                          goto LABEL_89;
                      }
                      if ( *v27 != -4096 )
                      {
                        v44 = v27[1] & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v44 )
                        {
                          if ( (*(_BYTE *)(v44 + 8) & 1) == 0 )
                          {
                            v76 = v30;
                            v84 = v27[1] & 0xFFFFFFFFFFFFFFF8LL;
                            sub_C7D6A0(*(_QWORD *)(v44 + 16), 16LL * *(unsigned int *)(v44 + 24), 8);
                            v30 = v76;
                            v44 = v84;
                          }
                          v81 = v30;
                          j_j___libc_free_0(v44, 272);
                          v30 = v81;
                        }
                        goto LABEL_57;
                      }
                      v27 += 2;
                      if ( v27 == v30 )
                      {
LABEL_89:
                        v59 = *(_DWORD *)(v13 + 304);
                        if ( v25 )
                        {
                          v60 = 64;
                          v61 = v25 - 1;
                          if ( v61 )
                          {
                            _BitScanReverse(&v62, v61);
                            v60 = 1 << (33 - (v62 ^ 0x1F));
                            if ( v60 < 64 )
                              v60 = 64;
                          }
                          v63 = *(_QWORD **)(v13 + 288);
                          if ( v59 == v60 )
                          {
                            *(_QWORD *)(v13 + 296) = 0;
                            v74 = &v63[2 * v59];
                            do
                            {
                              if ( v63 )
                                *v63 = -4096;
                              v63 += 2;
                            }
                            while ( v74 != v63 );
                          }
                          else
                          {
                            sub_C7D6A0((__int64)v63, v29, 8);
                            v64 = ((((((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v60 / 3u + 1)
                                     | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v60 / 3u + 1)
                                   | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v60 / 3u + 1)
                                   | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                 | (4 * v60 / 3u + 1)
                                 | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 16;
                            v65 = (v64
                                 | (((((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v60 / 3u + 1)
                                     | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v60 / 3u + 1)
                                   | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v60 / 3u + 1)
                                   | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1)) >> 2)
                                 | (4 * v60 / 3u + 1)
                                 | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1))
                                + 1;
                            *(_DWORD *)(v13 + 304) = v65;
                            v66 = (_QWORD *)sub_C7D670(16 * v65, 8);
                            v67 = *(unsigned int *)(v13 + 304);
                            *(_QWORD *)(v13 + 296) = 0;
                            *(_QWORD *)(v13 + 288) = v66;
                            for ( k = &v66[2 * v67]; k != v66; v66 += 2 )
                            {
                              if ( v66 )
                                *v66 = -4096;
                            }
                          }
LABEL_40:
                          v32 = *(_DWORD *)(v13 + 328);
                          ++*(_QWORD *)(v13 + 312);
                          if ( v32 )
                          {
                            v45 = 4 * v32;
                            v33 = *(unsigned int *)(v13 + 336);
                            if ( (unsigned int)(4 * v32) < 0x40 )
                              v45 = 64;
                            if ( v45 >= (unsigned int)v33 )
                            {
LABEL_43:
                              v34 = *(_QWORD **)(v13 + 320);
                              for ( m = &v34[2 * v33]; m != v34; v34 += 2 )
                                *v34 = -4096;
                              *(_QWORD *)(v13 + 328) = 0;
                              goto LABEL_46;
                            }
                            v46 = v32 - 1;
                            if ( v46 )
                            {
                              _BitScanReverse(&v46, v46);
                              v47 = *(_QWORD **)(v13 + 320);
                              v48 = 1 << (33 - (v46 ^ 0x1F));
                              if ( v48 < 64 )
                                v48 = 64;
                              if ( (_DWORD)v33 == v48 )
                              {
                                *(_QWORD *)(v13 + 328) = 0;
                                v49 = &v47[2 * (unsigned int)v33];
                                do
                                {
                                  if ( v47 )
                                    *v47 = -4096;
                                  v47 += 2;
                                }
                                while ( v49 != v47 );
                                goto LABEL_46;
                              }
                            }
                            else
                            {
                              v47 = *(_QWORD **)(v13 + 320);
                              v48 = 64;
                            }
                            sub_C7D6A0((__int64)v47, 16LL * (unsigned int)v33, 8);
                            v69 = ((((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v48 / 3u + 1)
                                     | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v48 / 3u + 1)
                                   | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v48 / 3u + 1)
                                   | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                 | (4 * v48 / 3u + 1)
                                 | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 16;
                            v70 = (v69
                                 | (((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v48 / 3u + 1)
                                     | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v48 / 3u + 1)
                                   | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v48 / 3u + 1)
                                   | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
                                 | (4 * v48 / 3u + 1)
                                 | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1))
                                + 1;
                            *(_DWORD *)(v13 + 336) = v70;
                            v71 = (_QWORD *)sub_C7D670(16 * v70, 8);
                            v72 = *(unsigned int *)(v13 + 336);
                            *(_QWORD *)(v13 + 328) = 0;
                            *(_QWORD *)(v13 + 320) = v71;
                            for ( n = &v71[2 * v72]; n != v71; v71 += 2 )
                            {
                              if ( v71 )
                                *v71 = -4096;
                            }
                          }
                          else if ( *(_DWORD *)(v13 + 332) )
                          {
                            v33 = *(unsigned int *)(v13 + 336);
                            if ( (unsigned int)v33 <= 0x40 )
                              goto LABEL_43;
                            sub_C7D6A0(*(_QWORD *)(v13 + 320), 16LL * (unsigned int)v33, 8);
                            *(_QWORD *)(v13 + 320) = 0;
                            *(_QWORD *)(v13 + 328) = 0;
                            *(_DWORD *)(v13 + 336) = 0;
                          }
LABEL_46:
                          v36 = *(_QWORD **)(v13 + 344);
                          while ( (_QWORD *)(v13 + 344) != v36 )
                          {
                            v37 = v36;
                            v36 = (_QWORD *)*v36;
                            v38 = v37[5];
                            v37[2] = &unk_49DB368;
                            if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
                            {
                              v80 = v37;
                              sub_BD60C0(v37 + 3);
                              v37 = v80;
                            }
                            j_j___libc_free_0(v37, 64);
                          }
                          *(_QWORD *)(v13 + 352) = v36;
                          *(_QWORD *)(v13 + 344) = v36;
                          *(_QWORD *)(v13 + 360) = 0;
                          sub_D21290(v78, (__int64)v77);
                          sub_D1EE30(v78, a3, v39, v40, v41, v42);
                          sub_D1FEC0(v78, v77);
                          goto LABEL_52;
                        }
                        if ( v59 )
                        {
                          sub_C7D6A0(*(_QWORD *)(v13 + 288), v29, 8);
                          *(_QWORD *)(v13 + 288) = 0;
                          *(_QWORD *)(v13 + 296) = 0;
                          *(_DWORD *)(v13 + 304) = 0;
                          goto LABEL_40;
                        }
LABEL_39:
                        *(_QWORD *)(v13 + 296) = 0;
                        goto LABEL_40;
                      }
                    }
                  }
                }
                else
                {
                  v52 = *(_QWORD **)(v13 + 256);
                  v53 = 64;
                }
                sub_C7D6A0((__int64)v52, 16LL * (unsigned int)v22, 8);
                v54 = ((((((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                         | (4 * v53 / 3u + 1)
                         | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                       | (4 * v53 / 3u + 1)
                       | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 8)
                     | (((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                       | (4 * v53 / 3u + 1)
                       | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                     | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                     | (4 * v53 / 3u + 1)
                     | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 16;
                v55 = (v54
                     | (((((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                         | (4 * v53 / 3u + 1)
                         | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                       | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                       | (4 * v53 / 3u + 1)
                       | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 8)
                     | (((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                       | (4 * v53 / 3u + 1)
                       | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                     | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                     | (4 * v53 / 3u + 1)
                     | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1))
                    + 1;
                *(_DWORD *)(v13 + 272) = v55;
                v56 = (_QWORD *)sub_C7D670(16 * v55, 8);
                v57 = *(unsigned int *)(v13 + 272);
                *(_QWORD *)(v13 + 264) = 0;
                *(_QWORD *)(v13 + 256) = v56;
                for ( ii = &v56[2 * v57]; ii != v56; v56 += 2 )
                {
                  if ( v56 )
                    *v56 = -4096;
                }
                goto LABEL_27;
              }
              memset(*(void **)(v13 + 160), -1, 8 * v20);
            }
            *(_QWORD *)(v13 + 172) = 0;
            goto LABEL_21;
          }
          v14 = 0xFFFFFFFFLL;
          memset(*(void **)(v13 + 56), -1, 8 * v17);
        }
        *(_QWORD *)(v13 + 68) = 0;
        goto LABEL_15;
      }
    }
  }
LABEL_52:
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
