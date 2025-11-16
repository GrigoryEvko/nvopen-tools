// Function: sub_2A72020
// Address: 0x2a72020
//
__int64 __fastcall sub_2A72020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // eax
  unsigned int v14; // eax
  unsigned __int8 **v15; // rax
  unsigned __int8 *v16; // r12
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // r14
  _QWORD *v23; // r15
  unsigned int v24; // esi
  __int64 v25; // rax
  unsigned __int8 *v26; // rdi
  char v27; // al
  unsigned int v28; // eax
  int v29; // eax
  int v30; // esi
  int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // r13
  __int64 v37; // rsi
  __int64 *v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r15
  _QWORD *j; // rdx
  __int64 k; // r13
  __int64 v47; // rax
  int v48; // esi
  int v49; // esi
  __int64 v50; // r9
  int v51; // r11d
  __int64 *v52; // r10
  unsigned int v53; // ecx
  __int64 *v54; // rdx
  __int64 v55; // rdi
  unsigned __int8 v56; // al
  int v57; // eax
  unsigned int v58; // eax
  _QWORD *v59; // rax
  __int64 v60; // r11
  __int64 v61; // r8
  _QWORD *i; // rdx
  __int64 v63; // r15
  __int64 v64; // rax
  int v65; // esi
  int v66; // esi
  __int64 v67; // r9
  unsigned int v68; // ecx
  __int64 *v69; // rdx
  __int64 v70; // rdi
  unsigned __int8 v71; // al
  int v72; // eax
  int v73; // ecx
  __int64 v74; // rdi
  int v75; // ecx
  _QWORD *v76; // rdi
  unsigned int v77; // r13d
  unsigned __int64 v78; // rdi
  unsigned __int64 v79; // rdi
  unsigned __int64 v80; // rdi
  unsigned __int64 v81; // rdi
  __int64 v82; // rdx
  _QWORD *v83; // rsi
  _QWORD *v84; // rdx
  __int64 v85; // rdx
  _QWORD *v86; // rsi
  _QWORD *v87; // rdx
  __int64 *v88; // r10
  __int64 v89; // [rsp+0h] [rbp-70h]
  __int64 v90; // [rsp+0h] [rbp-70h]
  unsigned int v91; // [rsp+8h] [rbp-68h]
  unsigned int v92; // [rsp+8h] [rbp-68h]
  __int64 v93; // [rsp+8h] [rbp-68h]
  __int64 v94; // [rsp+8h] [rbp-68h]
  __int64 v95; // [rsp+8h] [rbp-68h]
  __int64 v96; // [rsp+8h] [rbp-68h]
  int v97; // [rsp+8h] [rbp-68h]
  unsigned __int64 v98; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v99; // [rsp+18h] [rbp-58h]
  unsigned __int64 v100; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v101; // [rsp+28h] [rbp-48h]
  unsigned __int64 v102; // [rsp+30h] [rbp-40h]
  unsigned int v103; // [rsp+38h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 1984);
  result = *(unsigned int *)(a1 + 848);
  if ( !(_DWORD)v7 )
    goto LABEL_63;
  while ( 1 )
  {
    do
    {
      if ( (_DWORD)result )
        goto LABEL_3;
LABEL_8:
      v15 = *(unsigned __int8 ***)(a1 + 1384);
      if ( *(unsigned __int8 ***)(a1 + 1416) != v15 )
      {
LABEL_9:
        v16 = *v15;
        v17 = *(_QWORD *)(a1 + 1400) - 8LL;
        if ( v15 == (unsigned __int8 **)v17 )
        {
          j_j___libc_free_0(*(_QWORD *)(a1 + 1392));
          v38 = (__int64 *)(*(_QWORD *)(a1 + 1408) + 8LL);
          *(_QWORD *)(a1 + 1408) = v38;
          v39 = *v38;
          v17 = *v38 + 512;
          *(_QWORD *)(a1 + 1392) = v39;
          *(_QWORD *)(a1 + 1400) = v17;
          *(_QWORD *)(a1 + 1384) = v39;
        }
        else
        {
          *(_QWORD *)(a1 + 1384) = v15 + 1;
        }
        goto LABEL_11;
      }
      while ( 1 )
      {
        v28 = *(_DWORD *)(a1 + 1456);
        if ( !v28 )
          break;
        v17 = *(_QWORD *)(a1 + 1448);
        v7 = v28;
        v16 = *(unsigned __int8 **)(v17 + 8LL * v28 - 8);
        *(_DWORD *)(a1 + 1456) = v28 - 1;
LABEL_11:
        v18 = *(_DWORD *)(a1 + 352);
        v19 = *(_QWORD *)(a1 + 336);
        if ( v18 )
        {
          v17 = (unsigned int)(v18 - 1);
          v20 = v17 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v7 = v19 + 8LL * v20;
          v21 = *(_QWORD *)v7;
          if ( *(unsigned __int8 **)v7 == v16 )
          {
LABEL_13:
            *(_QWORD *)v7 = -8192;
            --*(_DWORD *)(a1 + 344);
            ++*(_DWORD *)(a1 + 348);
          }
          else
          {
            v7 = 1;
            while ( v21 != -4096 )
            {
              a5 = (unsigned int)(v7 + 1);
              v20 = v17 & (v7 + v20);
              v7 = v19 + 8LL * v20;
              v21 = *(_QWORD *)v7;
              if ( *(unsigned __int8 **)v7 == v16 )
                goto LABEL_13;
              v7 = (unsigned int)a5;
            }
          }
        }
        if ( *(_BYTE *)(*((_QWORD *)v16 + 1) + 8LL) == 15 )
          goto LABEL_19;
        v17 = *(unsigned int *)(a1 + 160);
        v22 = *(_QWORD *)(a1 + 144);
        if ( (_DWORD)v17 )
        {
          v7 = (unsigned int)(v17 - 1);
          a5 = 1;
          v23 = 0;
          v24 = v7 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v25 = v22 + 48LL * v24;
          v26 = *(unsigned __int8 **)v25;
          if ( *(unsigned __int8 **)v25 == v16 )
          {
LABEL_17:
            v27 = *(_BYTE *)(v25 + 8);
LABEL_18:
            if ( v27 == 6 )
              goto LABEL_8;
            goto LABEL_19;
          }
          while ( v26 != (unsigned __int8 *)-4096LL )
          {
            if ( !v23 && v26 == (unsigned __int8 *)-8192LL )
              v23 = (_QWORD *)v25;
            a6 = (unsigned int)(a5 + 1);
            v24 = v7 & (a5 + v24);
            v25 = v22 + 48LL * v24;
            v26 = *(unsigned __int8 **)v25;
            if ( *(unsigned __int8 **)v25 == v16 )
              goto LABEL_17;
            a5 = (unsigned int)a6;
          }
          if ( !v23 )
            v23 = (_QWORD *)v25;
          v29 = *(_DWORD *)(a1 + 152);
          ++*(_QWORD *)(a1 + 136);
          v30 = v29 + 1;
          if ( 4 * (v29 + 1) < (unsigned int)(3 * v17) )
          {
            if ( (int)v17 - *(_DWORD *)(a1 + 156) - v30 <= (unsigned int)v17 >> 3 )
            {
              v92 = v17;
              v58 = ((((((((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4) | (v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 8)
                     | (((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4)
                     | (v7 >> 1)
                     | v7
                     | (((v7 >> 1) | v7) >> 2)) >> 16)
                   | (((((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4) | (v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 8)
                   | (((v7 >> 1) | v7 | (((v7 >> 1) | v7) >> 2)) >> 4)
                   | (v7 >> 1)
                   | v7
                   | (((v7 >> 1) | v7) >> 2))
                  + 1;
              if ( v58 < 0x40 )
                v58 = 64;
              *(_DWORD *)(a1 + 160) = v58;
              v59 = (_QWORD *)sub_C7D670(48LL * v58, 8);
              *(_QWORD *)(a1 + 144) = v59;
              if ( v22 )
              {
                *(_QWORD *)(a1 + 152) = 0;
                v60 = 48LL * v92;
                v61 = v22 + v60;
                for ( i = &v59[6 * *(unsigned int *)(a1 + 160)]; i != v59; v59 += 6 )
                {
                  if ( v59 )
                    *v59 = -4096;
                }
                v63 = v22;
                do
                {
                  v64 = *(_QWORD *)v63;
                  if ( *(_QWORD *)v63 != -8192 && v64 != -4096 )
                  {
                    v65 = *(_DWORD *)(a1 + 160);
                    if ( !v65 )
                    {
                      MEMORY[0] = *(_QWORD *)v63;
                      BUG();
                    }
                    v66 = v65 - 1;
                    v67 = *(_QWORD *)(a1 + 144);
                    v68 = v66 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                    v69 = (__int64 *)(v67 + 48LL * v68);
                    v70 = *v69;
                    if ( v64 != *v69 )
                    {
                      v97 = 1;
                      v88 = 0;
                      while ( v70 != -4096 )
                      {
                        if ( !v88 && v70 == -8192 )
                          v88 = v69;
                        v68 = v66 & (v97 + v68);
                        v69 = (__int64 *)(v67 + 48LL * v68);
                        v70 = *v69;
                        if ( v64 == *v69 )
                          goto LABEL_108;
                        ++v97;
                      }
                      if ( v88 )
                        v69 = v88;
                    }
LABEL_108:
                    *v69 = v64;
                    v71 = *(_BYTE *)(v63 + 8);
                    *((_WORD *)v69 + 4) = v71;
                    if ( v71 <= 3u )
                    {
                      if ( v71 > 1u )
                        v69[2] = *(_QWORD *)(v63 + 16);
                    }
                    else if ( (unsigned __int8)(v71 - 4) <= 1u )
                    {
                      *((_DWORD *)v69 + 6) = *(_DWORD *)(v63 + 24);
                      v69[2] = *(_QWORD *)(v63 + 16);
                      v72 = *(_DWORD *)(v63 + 40);
                      *(_DWORD *)(v63 + 24) = 0;
                      *((_DWORD *)v69 + 10) = v72;
                      v69[4] = *(_QWORD *)(v63 + 32);
                      LOBYTE(v72) = *(_BYTE *)(v63 + 9);
                      *(_DWORD *)(v63 + 40) = 0;
                      *((_BYTE *)v69 + 9) = v72;
                    }
                    *(_BYTE *)(v63 + 8) = 0;
                    ++*(_DWORD *)(a1 + 152);
                    if ( (unsigned int)*(unsigned __int8 *)(v63 + 8) - 4 <= 1 )
                    {
                      if ( *(_DWORD *)(v63 + 40) > 0x40u )
                      {
                        v80 = *(_QWORD *)(v63 + 32);
                        if ( v80 )
                        {
                          v89 = v60;
                          v95 = v61;
                          j_j___libc_free_0_0(v80);
                          v60 = v89;
                          v61 = v95;
                        }
                      }
                      if ( *(_DWORD *)(v63 + 24) > 0x40u )
                      {
                        v81 = *(_QWORD *)(v63 + 16);
                        if ( v81 )
                        {
                          v90 = v60;
                          v96 = v61;
                          j_j___libc_free_0_0(v81);
                          v60 = v90;
                          v61 = v96;
                        }
                      }
                    }
                  }
                  v63 += 48;
                }
                while ( v61 != v63 );
                sub_C7D6A0(v22, v60, 8);
                v59 = *(_QWORD **)(a1 + 144);
                v75 = *(_DWORD *)(a1 + 160);
                v30 = *(_DWORD *)(a1 + 152) + 1;
              }
              else
              {
                v85 = *(unsigned int *)(a1 + 160);
                *(_QWORD *)(a1 + 152) = 0;
                v75 = v85;
                v86 = &v59[6 * v85];
                if ( v59 != v86 )
                {
                  v87 = v59;
                  do
                  {
                    if ( v87 )
                      *v87 = -4096;
                    v87 += 6;
                  }
                  while ( v86 != v87 );
                }
                v30 = 1;
              }
              if ( !v75 )
                goto LABEL_181;
              v7 = (unsigned int)(v75 - 1);
              a5 = 1;
              v76 = 0;
              v77 = v7 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v23 = &v59[6 * v77];
              v17 = *v23;
              if ( (unsigned __int8 *)*v23 != v16 )
              {
                while ( v17 != -4096 )
                {
                  if ( !v76 && v17 == -8192 )
                    v76 = v23;
                  a6 = (unsigned int)(a5 + 1);
                  v77 = v7 & (a5 + v77);
                  v23 = &v59[6 * v77];
                  v17 = *v23;
                  if ( (unsigned __int8 *)*v23 == v16 )
                    goto LABEL_33;
                  a5 = (unsigned int)a6;
                }
                if ( v76 )
                  v23 = v76;
              }
            }
            goto LABEL_33;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 136);
        }
        v91 = v17;
        v40 = (((((((unsigned int)(2 * v17 - 1) | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 2)
                | (unsigned int)(2 * v17 - 1)
                | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 4)
              | (((unsigned int)(2 * v17 - 1) | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 2)
              | (unsigned int)(2 * v17 - 1)
              | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 8)
            | (((((unsigned int)(2 * v17 - 1) | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 2)
              | (unsigned int)(2 * v17 - 1)
              | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 4)
            | (((unsigned int)(2 * v17 - 1) | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1)) >> 2)
            | (unsigned int)(2 * v17 - 1)
            | ((unsigned __int64)(unsigned int)(2 * v17 - 1) >> 1);
        v41 = ((v40 >> 16) | v40) + 1;
        if ( (unsigned int)v41 < 0x40 )
          LODWORD(v41) = 64;
        *(_DWORD *)(a1 + 160) = v41;
        v42 = (_QWORD *)sub_C7D670(48LL * (unsigned int)v41, 8);
        *(_QWORD *)(a1 + 144) = v42;
        if ( v22 )
        {
          *(_QWORD *)(a1 + 152) = 0;
          v43 = 48LL * v91;
          v44 = v22 + v43;
          for ( j = &v42[6 * *(unsigned int *)(a1 + 160)]; j != v42; v42 += 6 )
          {
            if ( v42 )
              *v42 = -4096;
          }
          for ( k = v22; v44 != k; k += 48 )
          {
            v47 = *(_QWORD *)k;
            if ( *(_QWORD *)k != -8192 && v47 != -4096 )
            {
              v48 = *(_DWORD *)(a1 + 160);
              if ( !v48 )
              {
                MEMORY[0] = *(_QWORD *)k;
                BUG();
              }
              v49 = v48 - 1;
              v50 = *(_QWORD *)(a1 + 144);
              v51 = 1;
              v52 = 0;
              v53 = v49 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v54 = (__int64 *)(v50 + 48LL * v53);
              v55 = *v54;
              if ( *v54 != v47 )
              {
                while ( v55 != -4096 )
                {
                  if ( !v52 && v55 == -8192 )
                    v52 = v54;
                  v53 = v49 & (v51 + v53);
                  v54 = (__int64 *)(v50 + 48LL * v53);
                  v55 = *v54;
                  if ( v47 == *v54 )
                    goto LABEL_89;
                  ++v51;
                }
                if ( v52 )
                  v54 = v52;
              }
LABEL_89:
              *v54 = v47;
              v56 = *(_BYTE *)(k + 8);
              *((_WORD *)v54 + 4) = v56;
              if ( v56 <= 3u )
              {
                if ( v56 > 1u )
                  v54[2] = *(_QWORD *)(k + 16);
              }
              else if ( (unsigned __int8)(v56 - 4) <= 1u )
              {
                *((_DWORD *)v54 + 6) = *(_DWORD *)(k + 24);
                v54[2] = *(_QWORD *)(k + 16);
                v57 = *(_DWORD *)(k + 40);
                *(_DWORD *)(k + 24) = 0;
                *((_DWORD *)v54 + 10) = v57;
                v54[4] = *(_QWORD *)(k + 32);
                LOBYTE(v57) = *(_BYTE *)(k + 9);
                *(_DWORD *)(k + 40) = 0;
                *((_BYTE *)v54 + 9) = v57;
              }
              *(_BYTE *)(k + 8) = 0;
              ++*(_DWORD *)(a1 + 152);
              if ( (unsigned int)*(unsigned __int8 *)(k + 8) - 4 <= 1 )
              {
                if ( *(_DWORD *)(k + 40) > 0x40u )
                {
                  v78 = *(_QWORD *)(k + 32);
                  if ( v78 )
                  {
                    v93 = v43;
                    j_j___libc_free_0_0(v78);
                    v43 = v93;
                  }
                }
                if ( *(_DWORD *)(k + 24) > 0x40u )
                {
                  v79 = *(_QWORD *)(k + 16);
                  if ( v79 )
                  {
                    v94 = v43;
                    j_j___libc_free_0_0(v79);
                    v43 = v94;
                  }
                }
              }
            }
          }
          sub_C7D6A0(v22, v43, 8);
          v42 = *(_QWORD **)(a1 + 144);
          v73 = *(_DWORD *)(a1 + 160);
          v30 = *(_DWORD *)(a1 + 152) + 1;
        }
        else
        {
          v82 = *(unsigned int *)(a1 + 160);
          *(_QWORD *)(a1 + 152) = 0;
          v73 = v82;
          v83 = &v42[6 * v82];
          if ( v42 != v83 )
          {
            v84 = v42;
            do
            {
              if ( v84 )
                *v84 = -4096;
              v84 += 6;
            }
            while ( v83 != v84 );
          }
          v30 = 1;
        }
        if ( !v73 )
        {
LABEL_181:
          ++*(_DWORD *)(a1 + 152);
          BUG();
        }
        v7 = (unsigned int)(v73 - 1);
        v17 = (unsigned int)v7 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v23 = &v42[6 * v17];
        v74 = *v23;
        if ( (unsigned __int8 *)*v23 != v16 )
        {
          a6 = 1;
          a5 = 0;
          while ( v74 != -4096 )
          {
            if ( v74 == -8192 && !a5 )
              a5 = (__int64)v23;
            v17 = (unsigned int)v7 & ((_DWORD)a6 + (_DWORD)v17);
            v23 = &v42[6 * (unsigned int)v17];
            v74 = *v23;
            if ( (unsigned __int8 *)*v23 == v16 )
              goto LABEL_33;
            a6 = (unsigned int)(a6 + 1);
          }
          if ( a5 )
            v23 = (_QWORD *)a5;
        }
LABEL_33:
        *(_DWORD *)(a1 + 152) = v30;
        if ( *v23 != -4096 )
          --*(_DWORD *)(a1 + 156);
        *v23 = v16;
        *((_WORD *)v23 + 4) = 0;
        v31 = *v16;
        if ( (unsigned __int8)v31 <= 0x15u )
        {
          v32 = (unsigned int)(v31 - 12);
          if ( (unsigned __int8)(v31 - 12) <= 1u )
          {
            *((_BYTE *)v23 + 8) = 1;
            sub_2A71D80(a1, (__int64)v16, v32, v7, a5, a6);
            goto LABEL_20;
          }
          if ( (_BYTE)v31 != 17 )
          {
            *((_BYTE *)v23 + 8) = 2;
            v23[2] = v16;
            sub_2A71D80(a1, (__int64)v16, v32, v7, a5, a6);
            goto LABEL_20;
          }
          v99 = *((_DWORD *)v16 + 8);
          if ( v99 > 0x40 )
            sub_C43780((__int64)&v98, (const void **)v16 + 3);
          else
            v98 = *((_QWORD *)v16 + 3);
          sub_AADBC0((__int64)&v100, (__int64 *)&v98);
          sub_2A62120((char *)v23 + 8, (__int64)&v100, 0, 0, 1u);
          if ( v103 > 0x40 && v102 )
            j_j___libc_free_0_0(v102);
          if ( v101 > 0x40 && v100 )
            j_j___libc_free_0_0(v100);
          if ( v99 > 0x40 && v98 )
            j_j___libc_free_0_0(v98);
          v27 = *((_BYTE *)v23 + 8);
          goto LABEL_18;
        }
LABEL_19:
        sub_2A71D80(a1, (__int64)v16, v17, v7, a5, a6);
LABEL_20:
        v15 = *(unsigned __int8 ***)(a1 + 1384);
        if ( *(unsigned __int8 ***)(a1 + 1416) != v15 )
          goto LABEL_9;
      }
LABEL_38:
      v33 = *(_DWORD *)(a1 + 1984);
      while ( v33 )
      {
        v7 = v33--;
        v34 = *(_QWORD *)(*(_QWORD *)(a1 + 1976) + 8 * v7 - 8);
        *(_DWORD *)(a1 + 1984) = v33;
        v35 = *(_QWORD *)(v34 + 56);
        v36 = v34 + 48;
        if ( v34 + 48 != v35 )
        {
          do
          {
            v37 = v35;
            v35 = *(_QWORD *)(v35 + 8);
            sub_2A71C10((unsigned __int64 *)a1, (unsigned __int8 *)(v37 - 24), v34, v7);
          }
          while ( v36 != v35 );
          goto LABEL_38;
        }
      }
      result = *(unsigned int *)(a1 + 848);
LABEL_63:
      ;
    }
    while ( *(_DWORD *)(a1 + 1456) || *(_QWORD *)(a1 + 1384) != *(_QWORD *)(a1 + 1416) );
    if ( !(_DWORD)result )
      return result;
LABEL_3:
    v9 = *(_QWORD *)(a1 + 840);
    v10 = (unsigned int)result;
    v11 = *(_QWORD *)(a1 + 336);
    v12 = *(_QWORD *)(v9 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a1 + 848) = result - 1;
    v13 = *(_DWORD *)(a1 + 352);
    if ( v13 )
    {
      v9 = (unsigned int)(v13 - 1);
      v14 = v9 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = v11 + 8LL * v14;
      a5 = *(_QWORD *)v10;
      if ( v12 == *(_QWORD *)v10 )
      {
LABEL_5:
        *(_QWORD *)v10 = -8192;
        --*(_DWORD *)(a1 + 344);
        ++*(_DWORD *)(a1 + 348);
      }
      else
      {
        v10 = 1;
        while ( a5 != -4096 )
        {
          a6 = (unsigned int)(v10 + 1);
          v14 = v9 & (v10 + v14);
          v10 = v11 + 8LL * v14;
          a5 = *(_QWORD *)v10;
          if ( v12 == *(_QWORD *)v10 )
            goto LABEL_5;
          v10 = (unsigned int)a6;
        }
      }
    }
    sub_2A71D80(a1, v12, v9, v10, a5, a6);
    LODWORD(result) = *(_DWORD *)(a1 + 848);
  }
}
