// Function: sub_2CEF6B0
// Address: 0x2cef6b0
//
void __fastcall sub_2CEF6B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // rsi
  unsigned int v9; // r13d
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // rbx
  unsigned __int64 v22; // r13
  __int64 v23; // r12
  char v24; // al
  char *v25; // r15
  unsigned int v26; // eax
  unsigned __int8 v27; // dl
  unsigned __int8 *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r11
  unsigned __int8 *v31; // r14
  unsigned int v32; // esi
  __int64 v33; // r9
  __int64 v34; // r8
  unsigned int v35; // edi
  __int64 v36; // r12
  unsigned __int8 *v37; // rcx
  __int64 v38; // rdx
  __m128i **v39; // rax
  unsigned __int64 v40; // r14
  unsigned __int64 v41; // r8
  const __m128i *v42; // rcx
  __m128i *v43; // rdx
  __int64 v44; // rdx
  unsigned __int8 **v45; // rax
  int v46; // ecx
  int v47; // ecx
  int v48; // esi
  int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // edx
  unsigned __int8 *v52; // rdi
  int v53; // r12d
  unsigned __int8 **v54; // r9
  int v55; // esi
  int v56; // esi
  __int64 v57; // r8
  int v58; // r12d
  unsigned int v59; // edx
  unsigned __int8 *v60; // rdi
  __int64 v61; // rax
  int v62; // eax
  bool v63; // al
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // r15
  __int64 v67; // rax
  _BYTE *v68; // rax
  unsigned __int8 *v69; // rcx
  int v70; // eax
  __int64 v71; // rcx
  __int64 v72; // r13
  __int64 v73; // rcx
  __int64 v74; // r12
  __int64 v75; // r15
  char v76; // al
  const char *v77; // rax
  __int64 v78; // rdx
  unsigned int v79; // ebx
  unsigned __int8 *v80; // r12
  unsigned __int8 v81; // cl
  __int64 v82; // rax
  unsigned int v83; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v84; // [rsp+Fh] [rbp-91h]
  __int64 v85; // [rsp+10h] [rbp-90h]
  int v86; // [rsp+10h] [rbp-90h]
  __int64 v87; // [rsp+10h] [rbp-90h]
  __int64 v88; // [rsp+10h] [rbp-90h]
  unsigned __int8 v90; // [rsp+18h] [rbp-88h]
  int v91; // [rsp+18h] [rbp-88h]
  unsigned __int64 v92; // [rsp+20h] [rbp-80h]
  int v93; // [rsp+20h] [rbp-80h]
  __int64 v94; // [rsp+28h] [rbp-78h]
  unsigned __int64 v95; // [rsp+28h] [rbp-78h]
  __int64 v96; // [rsp+28h] [rbp-78h]
  unsigned __int8 v97; // [rsp+28h] [rbp-78h]
  int v98; // [rsp+30h] [rbp-70h]
  unsigned __int8 v99; // [rsp+30h] [rbp-70h]
  int v100; // [rsp+30h] [rbp-70h]
  __int64 v101; // [rsp+30h] [rbp-70h]
  __int64 v102; // [rsp+30h] [rbp-70h]
  unsigned __int64 v103; // [rsp+38h] [rbp-68h]
  __int64 v104; // [rsp+38h] [rbp-68h]
  unsigned __int8 v105; // [rsp+38h] [rbp-68h]
  unsigned __int8 v106; // [rsp+38h] [rbp-68h]
  __int64 v107; // [rsp+38h] [rbp-68h]
  __int64 v108; // [rsp+38h] [rbp-68h]
  __int64 v109; // [rsp+48h] [rbp-58h] BYREF
  __int64 v110; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v111; // [rsp+58h] [rbp-48h]
  __int64 v112; // [rsp+60h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 360) )
  {
    v84 = qword_5014428;
    if ( (_BYTE)qword_5014428 )
    {
      v6 = *(_QWORD *)(a1 + 368);
      v109 = a1;
      if ( !*(_BYTE *)(v6 + 192) )
        sub_CFDFC0(v6, a2, a3, a4, a5, a6);
      v7 = *(_QWORD *)(v6 + 16);
      v103 = v7;
      v92 = v7 + 32LL * *(unsigned int *)(v6 + 24);
      if ( v92 == v7 )
        goto LABEL_20;
      while ( 1 )
      {
        v9 = 0;
        v10 = *(_QWORD *)(v103 + 16);
        if ( !v10 )
          goto LABEL_19;
        while ( *(char *)(v10 + 7) < 0 )
        {
          v11 = sub_BD2BC0(v10);
          v13 = v11 + v12;
          v14 = 0;
          if ( *(char *)(v10 + 7) < 0 )
            v14 = sub_BD2BC0(v10);
          if ( v9 >= (unsigned int)((v13 - v14) >> 4) )
            break;
          v15 = (_QWORD *)sub_BD5C60(v10);
          sub_BCB2E0(v15);
          v16 = 0;
          if ( *(char *)(v10 + 7) < 0 )
            v16 = sub_BD2BC0(v10);
          v17 = v16 + 16LL * v9;
          v18 = *(_QWORD **)v17;
          v19 = *(unsigned int *)(v17 + 12);
          if ( **(_QWORD **)v17 == 5 && *((_DWORD *)v18 + 4) == 1734962273 && *((_BYTE *)v18 + 20) == 110 )
          {
            v85 = 32LL * *(unsigned int *)(v17 + 8);
            v94 = v10 + v85 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
            v28 = sub_BD3990(*(unsigned __int8 **)v94, v7);
            v29 = v94;
            v30 = 0;
            v31 = v28;
            v95 = *(_QWORD *)(v94 + 32);
            if ( 32 * v19 - v85 == 96 )
              v30 = *(_QWORD *)(v29 + 64);
            v32 = *(_DWORD *)(a1 + 248);
            if ( v32 )
            {
              v33 = v32 - 1;
              v34 = *(_QWORD *)(a1 + 232);
              v35 = v33 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
              v36 = v34 + 48LL * v35;
              v37 = *(unsigned __int8 **)v36;
              if ( v28 == *(unsigned __int8 **)v36 )
              {
LABEL_46:
                v38 = *(unsigned int *)(v36 + 16);
                v110 = v10;
                v39 = (__m128i **)(v36 + 8);
                v40 = *(_QWORD *)(v36 + 8);
                v7 = *(unsigned int *)(v36 + 20);
                v112 = v30;
                v41 = v38 + 1;
                v111 = v95;
                v42 = (const __m128i *)&v110;
                v43 = (__m128i *)(v40 + 24 * v38);
                if ( v7 < v41 )
                {
                  v7 = v36 + 24;
                  if ( v40 > (unsigned __int64)&v110 || v43 <= (__m128i *)&v110 )
                  {
                    sub_C8D5F0(v36 + 8, (const void *)v7, v41, 0x18u, v41, v33);
                    v39 = (__m128i **)(v36 + 8);
                    v43 = (__m128i *)(*(_QWORD *)(v36 + 8) + 24LL * *(unsigned int *)(v36 + 16));
                    v42 = (const __m128i *)&v110;
                  }
                  else
                  {
                    sub_C8D5F0(v36 + 8, (const void *)v7, v41, 0x18u, v41, v33);
                    v44 = *(_QWORD *)(v36 + 8);
                    v39 = (__m128i **)(v36 + 8);
                    v7 = 3LL * *(unsigned int *)(v36 + 16);
                    v42 = (const __m128i *)((char *)&v110 + v44 - v40);
                    v43 = (__m128i *)(v44 + 24LL * *(unsigned int *)(v36 + 16));
                  }
                }
LABEL_47:
                *v43 = _mm_loadu_si128(v42);
                v43[1].m128i_i64[0] = v42[1].m128i_i64[0];
                ++*((_DWORD *)v39 + 2);
                goto LABEL_16;
              }
              v86 = 1;
              v45 = 0;
              while ( v37 != (unsigned __int8 *)-4096LL )
              {
                if ( v37 == (unsigned __int8 *)-8192LL && !v45 )
                  v45 = (unsigned __int8 **)v36;
                v35 = v33 & (v86 + v35);
                v36 = v34 + 48LL * v35;
                v37 = *(unsigned __int8 **)v36;
                if ( v31 == *(unsigned __int8 **)v36 )
                  goto LABEL_46;
                ++v86;
              }
              v46 = *(_DWORD *)(a1 + 240);
              if ( !v45 )
                v45 = (unsigned __int8 **)v36;
              ++*(_QWORD *)(a1 + 224);
              v47 = v46 + 1;
              if ( 4 * v47 < 3 * v32 )
              {
                if ( v32 - *(_DWORD *)(a1 + 244) - v47 <= v32 >> 3 )
                {
                  v83 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
                  v88 = v30;
                  sub_2CE6D00(a1 + 224, v32);
                  v55 = *(_DWORD *)(a1 + 248);
                  if ( !v55 )
                  {
LABEL_148:
                    ++*(_DWORD *)(a1 + 240);
                    BUG();
                  }
                  v56 = v55 - 1;
                  v54 = 0;
                  v57 = *(_QWORD *)(a1 + 232);
                  v30 = v88;
                  v58 = 1;
                  v59 = v56 & v83;
                  v47 = *(_DWORD *)(a1 + 240) + 1;
                  v45 = (unsigned __int8 **)(v57 + 48LL * (v56 & v83));
                  v60 = *v45;
                  if ( v31 != *v45 )
                  {
                    while ( v60 != (unsigned __int8 *)-4096LL )
                    {
                      if ( !v54 && v60 == (unsigned __int8 *)-8192LL )
                        v54 = v45;
                      v59 = v56 & (v58 + v59);
                      v45 = (unsigned __int8 **)(v57 + 48LL * v59);
                      v60 = *v45;
                      if ( v31 == *v45 )
                        goto LABEL_58;
                      ++v58;
                    }
                    goto LABEL_66;
                  }
                }
                goto LABEL_58;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 224);
            }
            v87 = v30;
            sub_2CE6D00(a1 + 224, 2 * v32);
            v48 = *(_DWORD *)(a1 + 248);
            if ( !v48 )
              goto LABEL_148;
            v49 = v48 - 1;
            v50 = *(_QWORD *)(a1 + 232);
            v30 = v87;
            v51 = v49 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v47 = *(_DWORD *)(a1 + 240) + 1;
            v45 = (unsigned __int8 **)(v50 + 48LL * v51);
            v52 = *v45;
            if ( v31 != *v45 )
            {
              v53 = 1;
              v54 = 0;
              while ( v52 != (unsigned __int8 *)-4096LL )
              {
                if ( v52 == (unsigned __int8 *)-8192LL && !v54 )
                  v54 = v45;
                v51 = v49 & (v53 + v51);
                v45 = (unsigned __int8 **)(v50 + 48LL * v51);
                v52 = *v45;
                if ( v31 == *v45 )
                  goto LABEL_58;
                ++v53;
              }
LABEL_66:
              if ( v54 )
                v45 = v54;
            }
LABEL_58:
            *(_DWORD *)(a1 + 240) = v47;
            if ( *v45 != (unsigned __int8 *)-4096LL )
              --*(_DWORD *)(a1 + 244);
            v43 = (__m128i *)(v45 + 3);
            *v45 = v31;
            v42 = (const __m128i *)&v110;
            v39 = (__m128i **)(v45 + 1);
            *v39 = v43;
            v39[1] = (__m128i *)0x100000000LL;
            v7 = v95;
            v110 = v10;
            v111 = v95;
            v112 = v30;
            goto LABEL_47;
          }
LABEL_16:
          ++v9;
        }
        v7 = *(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)v7 == 85 )
        {
          v61 = *(_QWORD *)(v7 - 32);
          if ( v61 )
          {
            if ( !*(_BYTE *)v61 && *(_QWORD *)(v61 + 24) == *(_QWORD *)(v7 + 80) && (*(_BYTE *)(v61 + 33) & 0x20) != 0 )
              sub_2CEEF80(&v109, v7, *(_QWORD *)(v103 + 16));
          }
        }
LABEL_19:
        v103 += 32LL;
        if ( v92 == v103 )
        {
LABEL_20:
          v20 = *(_QWORD *)(a2 + 80);
          v21 = a2 + 72;
          if ( v20 == a2 + 72 )
            return;
          while ( 2 )
          {
            if ( !v20 )
              BUG();
            v22 = *(_QWORD *)(v20 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v22 == v20 + 24 )
              goto LABEL_149;
            if ( !v22 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 > 0xA )
LABEL_149:
              BUG();
            if ( *(_BYTE *)(v22 - 24) == 31 && (*(_DWORD *)(v22 - 20) & 0x7FFFFFF) == 3 )
            {
              v23 = *(_QWORD *)(v22 - 120);
              v24 = *(_BYTE *)v23;
              if ( *(_BYTE *)v23 > 0x1Cu )
              {
                if ( v24 != 68 )
                  goto LABEL_31;
                v23 = *(_QWORD *)(v23 - 32);
                if ( v23 )
                {
                  v24 = *(_BYTE *)v23;
                  if ( *(_BYTE *)v23 <= 0x1Cu )
                    BUG();
LABEL_31:
                  if ( v24 != 82 )
                  {
LABEL_37:
                    v27 = 0;
                    goto LABEL_38;
                  }
                  v25 = *(char **)(v23 - 64);
                  if ( v25 )
                  {
                    if ( **(_BYTE **)(v23 - 32) <= 0x15u )
                    {
                      v104 = *(_QWORD *)(v23 - 32);
                      LOBYTE(v26) = sub_AC30F0(v104);
                      v27 = v26;
                      if ( (_BYTE)v26 )
                        goto LABEL_35;
                      if ( *(_BYTE *)v104 == 17 )
                      {
                        if ( *(_DWORD *)(v104 + 32) <= 0x40u )
                        {
                          v63 = *(_QWORD *)(v104 + 24) == 0;
                        }
                        else
                        {
                          v98 = *(_DWORD *)(v104 + 32);
                          v62 = sub_C444A0(v104 + 24);
                          v27 = 0;
                          v63 = v98 == v62;
                        }
LABEL_85:
                        if ( !v63 )
                          goto LABEL_86;
LABEL_35:
                        if ( (unsigned int)sub_B53900(v23) != 32 )
                        {
                          v24 = *(_BYTE *)v23;
                          goto LABEL_37;
                        }
                        v24 = *v25;
                        if ( (unsigned __int8)*v25 <= 0x1Cu )
                          goto LABEL_39;
                        v27 = v84;
                        v23 = (__int64)v25;
LABEL_38:
                        if ( v24 != 85 )
                          goto LABEL_39;
                      }
                      else
                      {
                        v96 = *(_QWORD *)(v104 + 8);
                        if ( (unsigned int)*(unsigned __int8 *)(v96 + 8) - 17 > 1 )
                          goto LABEL_86;
                        v99 = v26;
                        v68 = sub_AD7630(v104, 0, v26);
                        v69 = (unsigned __int8 *)v104;
                        v27 = v99;
                        if ( !v68 || *v68 != 17 )
                        {
                          if ( *(_BYTE *)(v96 + 8) == 17 )
                          {
                            v93 = *(_DWORD *)(v96 + 32);
                            if ( v93 )
                            {
                              v97 = v99;
                              v108 = v21;
                              v79 = 0;
                              v102 = v23;
                              v80 = v69;
                              v81 = 0;
                              do
                              {
                                v90 = v81;
                                v82 = sub_AD69F0(v80, v79);
                                if ( !v82 )
                                {
LABEL_137:
                                  v21 = v108;
                                  v23 = v102;
                                  v27 = v97;
                                  goto LABEL_86;
                                }
                                v81 = v90;
                                if ( *(_BYTE *)v82 != 13 )
                                {
                                  if ( *(_BYTE *)v82 != 17 )
                                    goto LABEL_137;
                                  if ( *(_DWORD *)(v82 + 32) <= 0x40u )
                                  {
                                    if ( *(_QWORD *)(v82 + 24) )
                                      goto LABEL_137;
                                  }
                                  else
                                  {
                                    v91 = *(_DWORD *)(v82 + 32);
                                    if ( v91 != (unsigned int)sub_C444A0(v82 + 24) )
                                      goto LABEL_137;
                                  }
                                  v81 = v84;
                                }
                                ++v79;
                              }
                              while ( v93 != v79 );
                              v21 = v108;
                              v23 = v102;
                              v27 = v97;
                              if ( v81 )
                                goto LABEL_35;
                            }
                          }
                          goto LABEL_86;
                        }
                        if ( *((_DWORD *)v68 + 8) > 0x40u )
                        {
                          v100 = *((_DWORD *)v68 + 8);
                          v106 = v27;
                          v70 = sub_C444A0((__int64)(v68 + 24));
                          v27 = v106;
                          v63 = v100 == v70;
                          goto LABEL_85;
                        }
                        if ( !*((_QWORD *)v68 + 3) )
                          goto LABEL_35;
LABEL_86:
                        if ( *(_BYTE *)v23 != 85 )
                          goto LABEL_39;
                      }
                      v64 = *(_QWORD *)(v23 - 32);
                      if ( v64 )
                      {
                        if ( !*(_BYTE *)v64
                          && *(_QWORD *)(v64 + 24) == *(_QWORD *)(v23 + 80)
                          && (*(_BYTE *)(v64 + 33) & 0x20) != 0 )
                        {
                          v105 = v27;
                          v65 = 32LL * v27;
                          v66 = *(_QWORD *)(v22 + -32 - v65 - 24);
                          if ( v66 )
                          {
                            if ( sub_AA54C0(*(_QWORD *)(v22 + -32 - v65 - 24)) )
                            {
LABEL_93:
                              v67 = sub_AA4FF0(v66);
                              if ( v67 )
                                sub_2CEEF80(&v109, v23, v67 - 24);
                            }
                            else
                            {
                              v71 = *(_QWORD *)(v22 + -32 - 32LL * (v105 ^ 1u) - 24);
                              if ( v71 )
                              {
                                v72 = *(_QWORD *)(v71 + 56);
                                v73 = v71 + 48;
                                if ( v72 != v73 )
                                {
                                  v107 = v23;
                                  v74 = v73;
                                  v101 = v66;
                                  do
                                  {
                                    if ( !v72 )
                                      BUG();
                                    v76 = *(_BYTE *)(v72 - 24);
                                    if ( v76 == 85 )
                                    {
                                      v75 = *(_QWORD *)(v72 - 56);
                                      if ( v75 )
                                      {
                                        if ( !*(_BYTE *)v75 && *(_QWORD *)(v75 + 24) == *(_QWORD *)(v72 + 56) )
                                        {
                                          if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v72 - 56), 36)
                                            || (v77 = sub_BD5D20(v75), v78 == 12)
                                            && *(_QWORD *)v77 == 0x7472657373615F5FLL
                                            && *((_DWORD *)v77 + 2) == 1818845542 )
                                          {
LABEL_117:
                                            v23 = v107;
                                            v66 = v101;
                                            goto LABEL_93;
                                          }
                                        }
                                      }
                                    }
                                    else if ( v76 == 36 )
                                    {
                                      goto LABEL_117;
                                    }
                                    v72 = *(_QWORD *)(v72 + 8);
                                  }
                                  while ( v74 != v72 );
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
LABEL_39:
            v20 = *(_QWORD *)(v20 + 8);
            if ( v21 == v20 )
              return;
            continue;
          }
        }
      }
    }
  }
}
