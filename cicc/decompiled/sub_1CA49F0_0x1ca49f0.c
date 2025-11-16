// Function: sub_1CA49F0
// Address: 0x1ca49f0
//
void __fastcall sub_1CA49F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  __int64 v5; // r14
  __int64 v6; // rbx
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // r14
  __int64 v23; // r13
  unsigned __int8 v24; // al
  __int64 v25; // r15
  __int64 v26; // rcx
  unsigned int v27; // eax
  unsigned __int8 v28; // dl
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rcx
  __int64 v32; // r11
  unsigned int v33; // esi
  int v34; // r9d
  __int64 v35; // r8
  unsigned int v36; // edi
  __int64 *v37; // r13
  __int64 v38; // rcx
  __int64 v39; // rax
  unsigned int v40; // edx
  __m128i *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // rdx
  int v49; // ecx
  int v50; // ecx
  __int64 *v51; // r13
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  unsigned int v55; // edx
  __int64 v56; // rdi
  int v57; // r9d
  __int64 *v58; // r10
  int v59; // esi
  int v60; // esi
  __int64 v61; // r8
  int v62; // r9d
  unsigned int v63; // edx
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // r14
  __int64 v67; // r8
  const char *v68; // rax
  __int64 v69; // rdx
  char v70; // al
  __int64 v71; // rax
  __int64 v72; // rcx
  int v73; // eax
  __int64 v74; // rax
  char v75; // di
  int v76; // eax
  __int64 v77; // [rsp+8h] [rbp-98h]
  unsigned int v78; // [rsp+8h] [rbp-98h]
  int v79; // [rsp+10h] [rbp-90h]
  __int64 v80; // [rsp+10h] [rbp-90h]
  __int64 v81; // [rsp+10h] [rbp-90h]
  int v82; // [rsp+10h] [rbp-90h]
  char v83; // [rsp+1Fh] [rbp-81h]
  __int64 v84; // [rsp+20h] [rbp-80h]
  __int64 *v85; // [rsp+20h] [rbp-80h]
  __int64 v86; // [rsp+20h] [rbp-80h]
  __int64 v87; // [rsp+20h] [rbp-80h]
  unsigned __int8 v88; // [rsp+20h] [rbp-80h]
  __int64 *v90; // [rsp+30h] [rbp-70h]
  __int64 v91; // [rsp+30h] [rbp-70h]
  unsigned int v92; // [rsp+30h] [rbp-70h]
  __int64 v93; // [rsp+30h] [rbp-70h]
  unsigned __int8 v94; // [rsp+30h] [rbp-70h]
  int v95; // [rsp+30h] [rbp-70h]
  __int64 v96; // [rsp+30h] [rbp-70h]
  __int64 v97; // [rsp+38h] [rbp-68h]
  __int64 v98; // [rsp+38h] [rbp-68h]
  unsigned __int8 v99; // [rsp+38h] [rbp-68h]
  __int64 v100; // [rsp+38h] [rbp-68h]
  unsigned __int8 v101; // [rsp+38h] [rbp-68h]
  int v102; // [rsp+38h] [rbp-68h]
  __int64 v103; // [rsp+48h] [rbp-58h] BYREF
  __m128i v104; // [rsp+50h] [rbp-50h] BYREF
  __int64 v105; // [rsp+60h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 360) )
  {
    v83 = byte_4FBE380;
    if ( byte_4FBE380 )
    {
      v2 = *(_QWORD *)(a1 + 368);
      v103 = a1;
      if ( !*(_BYTE *)(v2 + 184) )
        sub_14CDF70(v2);
      v3 = *(_QWORD *)(v2 + 8);
      v5 = v3;
      v97 = v3 + 32LL * *(unsigned int *)(v2 + 16);
      if ( v97 == v3 )
        goto LABEL_19;
      while ( 1 )
      {
        v6 = *(_QWORD *)(v5 + 16);
        v7 = 0;
        if ( !v6 )
          goto LABEL_18;
        while ( *(char *)(v6 + 23) < 0 )
        {
          v8 = sub_1648A40(v6);
          v10 = v8 + v9;
          v11 = 0;
          if ( *(char *)(v6 + 23) < 0 )
            v11 = sub_1648A40(v6);
          if ( v7 >= (unsigned int)((v10 - v11) >> 4) )
            break;
          v12 = (_QWORD *)sub_16498A0(v6);
          sub_1643360(v12);
          v13 = 0;
          if ( *(char *)(v6 + 23) < 0 )
            v13 = sub_1648A40(v6);
          v14 = v13 + 16LL * v7;
          v15 = *(_QWORD **)v14;
          v16 = *(unsigned int *)(v14 + 8);
          v17 = *(unsigned int *)(v14 + 12);
          if ( **(_QWORD **)v14 == 5 && *((_DWORD *)v15 + 4) == 1734962273 && *((_BYTE *)v15 + 20) == 110 )
          {
            v84 = 24 * v16;
            v90 = (__int64 *)(v6 + 24 * v16 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
            v30 = sub_1649C60(*v90);
            v31 = v90;
            v32 = 0;
            v91 = v90[3];
            if ( 24 * v17 - 72 == v84 )
              v32 = v31[6];
            v33 = *(_DWORD *)(a1 + 248);
            if ( v33 )
            {
              v34 = v33 - 1;
              v35 = *(_QWORD *)(a1 + 232);
              v36 = (v33 - 1) & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
              v37 = (__int64 *)(v35 + 48LL * v36);
              v38 = *v37;
              if ( v30 == *v37 )
              {
LABEL_42:
                v39 = *((unsigned int *)v37 + 4);
                v40 = *((_DWORD *)v37 + 5);
                v104.m128i_i64[0] = v6;
                v105 = v32;
                v104.m128i_i64[1] = v91;
                if ( (unsigned int)v39 >= v40 )
                {
                  sub_16CD150((__int64)(v37 + 1), v37 + 3, 0, 24, v35, v34);
                  v39 = *((unsigned int *)v37 + 4);
                }
LABEL_44:
                v41 = (__m128i *)(v37[1] + 24 * v39);
                v42 = v105;
                *v41 = _mm_loadu_si128(&v104);
                v41[1].m128i_i64[0] = v42;
                ++*((_DWORD *)v37 + 4);
                goto LABEL_15;
              }
              v79 = 1;
              v85 = 0;
              v77 = *(_QWORD *)(a1 + 232);
              while ( v38 != -8 )
              {
                if ( v38 != -16 || v85 )
                  v37 = v85;
                v36 = v34 & (v79 + v36);
                LODWORD(v35) = v77 + 48 * v36;
                v38 = *(_QWORD *)(v77 + 48LL * v36);
                if ( v30 == v38 )
                {
                  v37 = (__int64 *)(v77 + 48LL * v36);
                  goto LABEL_42;
                }
                v85 = v37;
                v37 = (__int64 *)(v77 + 48LL * v36);
                ++v79;
              }
              if ( v85 )
                v37 = v85;
              v49 = *(_DWORD *)(a1 + 240);
              ++*(_QWORD *)(a1 + 224);
              v50 = v49 + 1;
              if ( 4 * v50 < 3 * v33 )
              {
                if ( v33 - *(_DWORD *)(a1 + 244) - v50 <= v33 >> 3 )
                {
                  v78 = ((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9);
                  v81 = v32;
                  v87 = v30;
                  sub_1CA0D00(a1 + 224, v33);
                  v59 = *(_DWORD *)(a1 + 248);
                  if ( !v59 )
                  {
LABEL_137:
                    ++*(_DWORD *)(a1 + 240);
                    BUG();
                  }
                  v60 = v59 - 1;
                  v58 = 0;
                  v61 = *(_QWORD *)(a1 + 232);
                  v32 = v81;
                  v62 = 1;
                  v63 = v60 & v78;
                  v50 = *(_DWORD *)(a1 + 240) + 1;
                  v30 = v87;
                  v37 = (__int64 *)(v61 + 48LL * (v60 & v78));
                  v64 = *v37;
                  if ( v87 != *v37 )
                  {
                    while ( v64 != -8 )
                    {
                      if ( v64 == -16 && !v58 )
                        v58 = v37;
                      v63 = v60 & (v62 + v63);
                      v37 = (__int64 *)(v61 + 48LL * v63);
                      v64 = *v37;
                      if ( v87 == *v37 )
                        goto LABEL_64;
                      ++v62;
                    }
                    goto LABEL_77;
                  }
                }
                goto LABEL_64;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 224);
            }
            v80 = v32;
            v86 = v30;
            sub_1CA0D00(a1 + 224, 2 * v33);
            v52 = *(_DWORD *)(a1 + 248);
            if ( !v52 )
              goto LABEL_137;
            v30 = v86;
            v53 = v52 - 1;
            v54 = *(_QWORD *)(a1 + 232);
            v32 = v80;
            v55 = v53 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
            v37 = (__int64 *)(v54 + 48LL * v55);
            v50 = *(_DWORD *)(a1 + 240) + 1;
            v56 = *v37;
            if ( v86 != *v37 )
            {
              v57 = 1;
              v58 = 0;
              while ( v56 != -8 )
              {
                if ( !v58 && v56 == -16 )
                  v58 = v37;
                v55 = v53 & (v57 + v55);
                v37 = (__int64 *)(v54 + 48LL * v55);
                v56 = *v37;
                if ( v86 == *v37 )
                  goto LABEL_64;
                ++v57;
              }
LABEL_77:
              if ( v58 )
                v37 = v58;
            }
LABEL_64:
            *(_DWORD *)(a1 + 240) = v50;
            if ( *v37 != -8 )
              --*(_DWORD *)(a1 + 244);
            *v37 = v30;
            v37[1] = (__int64)(v37 + 3);
            v37[2] = 0x100000000LL;
            v104.m128i_i64[0] = v6;
            v104.m128i_i64[1] = v91;
            v39 = 0;
            v105 = v32;
            goto LABEL_44;
          }
LABEL_15:
          ++v7;
        }
        v3 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v3 + 16) == 78 )
        {
          v43 = *(_QWORD *)(v3 - 24);
          if ( !*(_BYTE *)(v43 + 16) && (*(_BYTE *)(v43 + 33) & 0x20) != 0 )
            sub_1CA42C0(&v103, v3, *(_QWORD *)(v5 + 16));
        }
LABEL_18:
        v5 += 32;
        if ( v97 == v5 )
        {
LABEL_19:
          v18 = *(_QWORD *)(a2 + 80);
          if ( v18 == a2 + 72 )
            return;
          while ( 2 )
          {
            v19 = v18 - 24;
            if ( !v18 )
              v19 = 0;
            v20 = sub_157EBA0(v19);
            v22 = v20;
            if ( *(_BYTE *)(v20 + 16) != 26 )
              goto LABEL_34;
            if ( (*(_DWORD *)(v20 + 20) & 0xFFFFFFF) != 3 )
              goto LABEL_34;
            v23 = *(_QWORD *)(v20 - 72);
            v24 = *(_BYTE *)(v23 + 16);
            if ( v24 <= 0x17u )
              goto LABEL_34;
            if ( v24 != 61 )
              goto LABEL_26;
            v51 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
                ? *(__int64 **)(v23 - 8)
                : (__int64 *)(v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF));
            v23 = *v51;
            if ( !v23 )
              goto LABEL_34;
            v24 = *(_BYTE *)(v23 + 16);
            if ( v24 <= 0x17u )
              BUG();
LABEL_26:
            if ( v24 != 75 )
            {
LABEL_32:
              v28 = 0;
              goto LABEL_33;
            }
            v25 = *(_QWORD *)(v23 - 48);
            if ( !v25 )
              goto LABEL_34;
            v26 = *(_QWORD *)(v23 - 24);
            if ( *(_BYTE *)(v26 + 16) > 0x10u )
              goto LABEL_34;
            v98 = *(_QWORD *)(v23 - 24);
            LOBYTE(v27) = sub_1593BB0(v98, v3, v21, v26);
            v28 = v27;
            if ( (_BYTE)v27 )
              goto LABEL_30;
            if ( *(_BYTE *)(v98 + 16) == 13 )
            {
              v3 = *(unsigned int *)(v98 + 32);
              if ( (unsigned int)v3 > 0x40 )
              {
                v92 = *(_DWORD *)(v98 + 32);
                v44 = sub_16A57B0(v98 + 24);
                v3 = v92;
                v28 = 0;
                if ( v92 != v44 )
                  break;
LABEL_30:
                v29 = *(unsigned __int16 *)(v23 + 18);
                BYTE1(v29) &= ~0x80u;
                if ( v29 != 32 )
                {
                  v24 = *(_BYTE *)(v23 + 16);
                  goto LABEL_32;
                }
                v24 = *(_BYTE *)(v25 + 16);
                v28 = v83;
                v23 = v25;
                if ( v24 <= 0x17u )
                {
LABEL_34:
                  v18 = *(_QWORD *)(v18 + 8);
                  if ( a2 + 72 == v18 )
                    return;
                  continue;
                }
LABEL_33:
                if ( v24 != 78 )
                  goto LABEL_34;
LABEL_52:
                v45 = *(_QWORD *)(v23 - 24);
                if ( !*(_BYTE *)(v45 + 16) && (*(_BYTE *)(v45 + 33) & 0x20) != 0 )
                {
                  v99 = v28;
                  v46 = 24LL * v28;
                  v3 = -24 - v46;
                  v47 = *(_QWORD *)(v22 + -24 - v46);
                  if ( v47 )
                  {
                    if ( sub_157F0B0(*(_QWORD *)(v22 + -24 - v46)) )
                    {
LABEL_56:
                      v48 = sub_157ED20(v47);
                      if ( v48 )
                      {
                        v3 = v23;
                        sub_1CA42C0(&v103, v23, v48);
                      }
                    }
                    else
                    {
                      v65 = *(_QWORD *)(v22 + -24 - 24LL * (v99 ^ 1u));
                      if ( v65 )
                      {
                        v66 = *(_QWORD *)(v65 + 48);
                        v100 = v65 + 40;
                        if ( v66 != v65 + 40 )
                        {
                          while ( 1 )
                          {
                            if ( !v66 )
                              BUG();
                            v70 = *(_BYTE *)(v66 - 8);
                            if ( v70 == 78 )
                            {
                              v67 = *(_QWORD *)(v66 - 48);
                              if ( !*(_BYTE *)(v67 + 16) )
                              {
                                v3 = 29;
                                v93 = *(_QWORD *)(v66 - 48);
                                if ( (unsigned __int8)sub_1560180(v67 + 112, 29) )
                                  goto LABEL_56;
                                v68 = sub_1649960(v93);
                                if ( v69 == 12
                                  && *(_QWORD *)v68 == 0x7472657373615F5FLL
                                  && *((_DWORD *)v68 + 2) == 1818845542 )
                                {
                                  goto LABEL_56;
                                }
                              }
                            }
                            else if ( v70 == 31 )
                            {
                              goto LABEL_56;
                            }
                            v66 = *(_QWORD *)(v66 + 8);
                            if ( v100 == v66 )
                              goto LABEL_34;
                          }
                        }
                      }
                    }
                  }
                }
                goto LABEL_34;
              }
              if ( !*(_QWORD *)(v98 + 24) )
                goto LABEL_30;
            }
            else if ( *(_BYTE *)(*(_QWORD *)v98 + 8LL) == 16 )
            {
              v94 = v27;
              v71 = sub_15A1020((_BYTE *)v98, v3, v27, v98);
              v72 = v98;
              v28 = v94;
              if ( v71 && *(_BYTE *)(v71 + 16) == 13 )
              {
                if ( *(_DWORD *)(v71 + 32) <= 0x40u )
                {
                  if ( !*(_QWORD *)(v71 + 24) )
                    goto LABEL_30;
                }
                else
                {
                  v95 = *(_DWORD *)(v71 + 32);
                  v101 = v28;
                  v73 = sub_16A57B0(v71 + 24);
                  v28 = v101;
                  if ( v95 == v73 )
                    goto LABEL_30;
                }
              }
              else
              {
                v102 = *(_QWORD *)(*(_QWORD *)v98 + 32LL);
                if ( !v102 )
                  goto LABEL_30;
                LODWORD(v3) = 0;
                while ( 1 )
                {
                  v88 = v28;
                  v96 = v72;
                  v74 = sub_15A0A60(v72, v3);
                  v72 = v96;
                  v3 = (unsigned int)v3;
                  v28 = v88;
                  if ( !v74 )
                    break;
                  v75 = *(_BYTE *)(v74 + 16);
                  if ( v75 != 9 )
                  {
                    if ( v75 != 13 )
                      goto LABEL_51;
                    if ( *(_DWORD *)(v74 + 32) <= 0x40u )
                    {
                      if ( *(_QWORD *)(v74 + 24) )
                        goto LABEL_51;
                    }
                    else
                    {
                      v82 = *(_DWORD *)(v74 + 32);
                      v76 = sub_16A57B0(v74 + 24);
                      v3 = (unsigned int)v3;
                      v72 = v96;
                      v28 = v88;
                      if ( v82 != v76 )
                        goto LABEL_51;
                    }
                  }
                  v3 = (unsigned int)(v3 + 1);
                  if ( v102 == (_DWORD)v3 )
                    goto LABEL_30;
                }
              }
            }
            break;
          }
LABEL_51:
          if ( *(_BYTE *)(v23 + 16) != 78 )
            goto LABEL_34;
          goto LABEL_52;
        }
      }
    }
  }
}
