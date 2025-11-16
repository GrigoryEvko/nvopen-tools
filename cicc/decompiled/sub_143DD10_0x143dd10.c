// Function: sub_143DD10
// Address: 0x143dd10
//
void __fastcall sub_143DD10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  int *v11; // rdi
  int *v12; // r13
  int *v13; // r14
  unsigned int v14; // esi
  int v15; // r12d
  __int64 v16; // r8
  unsigned int v17; // edi
  int *v18; // rdx
  int v19; // ecx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rsi
  __int64 v23; // rcx
  _QWORD *v24; // rdx
  int v25; // eax
  int v26; // eax
  __int64 v27; // rcx
  int v28; // edi
  unsigned int v29; // edx
  int *v30; // r15
  int v31; // esi
  unsigned __int64 v32; // rdi
  int v33; // eax
  int v34; // edx
  __int64 v35; // rcx
  int v36; // edi
  unsigned int v37; // eax
  int *v38; // r15
  int v39; // esi
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rax
  int v42; // r8d
  int v43; // r8d
  __int64 v44; // r9
  unsigned int v45; // edi
  __int64 *v46; // rax
  __int64 v47; // r11
  __int64 v48; // rdx
  __int64 v49; // rcx
  int v50; // eax
  int v51; // r10d
  int v52; // r10d
  int *v53; // rax
  int v54; // edi
  int v55; // edx
  int v56; // r8d
  int v57; // r8d
  __int64 v58; // r9
  __int64 v59; // rcx
  int v60; // r11d
  int v61; // edi
  int *v62; // rsi
  int v63; // edi
  int v64; // edi
  __int64 v65; // r9
  int *v66; // rcx
  __int64 v67; // r15
  int v68; // esi
  int v69; // r8d
  _QWORD *v70; // rdx
  __int64 v71; // [rsp+18h] [rbp-68h]
  int *v72; // [rsp+20h] [rbp-60h] BYREF
  __int64 v73; // [rsp+28h] [rbp-58h]
  _BYTE v74[80]; // [rsp+30h] [rbp-50h] BYREF

  v72 = (int *)v74;
  v73 = 0x800000000LL;
  v71 = a1 + 72;
  if ( *(_DWORD *)(a1 + 88) )
  {
    v2 = *(_QWORD *)(a1 + 80);
    v4 = v2 + 80LL * *(unsigned int *)(a1 + 96);
    if ( v2 != v4 )
    {
      while ( 1 )
      {
        v5 = v2;
        if ( *(_DWORD *)v2 <= 0xFFFFFFFD )
          break;
        v2 += 80;
        if ( v4 == v2 )
          return;
      }
      if ( v2 != v4 )
      {
        v6 = *(_QWORD **)(v2 + 24);
        v7 = *(_QWORD **)(v2 + 16);
        if ( v6 == v7 )
          goto LABEL_53;
LABEL_9:
        v8 = &v6[*(unsigned int *)(v5 + 32)];
        v7 = (_QWORD *)sub_16CC9F0(v5 + 8, a2);
        if ( a2 == *v7 )
        {
          v48 = *(_QWORD *)(v5 + 24);
          if ( v48 == *(_QWORD *)(v5 + 16) )
            v49 = *(unsigned int *)(v5 + 36);
          else
            v49 = *(unsigned int *)(v5 + 32);
          v70 = (_QWORD *)(v48 + 8 * v49);
        }
        else
        {
          v9 = *(_QWORD *)(v5 + 24);
          if ( v9 != *(_QWORD *)(v5 + 16) )
          {
            v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(v5 + 32));
            goto LABEL_12;
          }
          v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(v5 + 36));
          v70 = v7;
        }
        while ( 1 )
        {
          while ( v70 != v7 && *v7 >= 0xFFFFFFFFFFFFFFFELL )
            ++v7;
LABEL_12:
          if ( v7 != v8 )
          {
            v10 = (unsigned int)v73;
            if ( (unsigned int)v73 >= HIDWORD(v73) )
            {
              sub_16CD150(&v72, v74, 0, 4);
              v10 = (unsigned int)v73;
            }
            v72[v10] = *(_DWORD *)v5;
            LODWORD(v73) = v73 + 1;
          }
          v5 += 80;
          if ( v5 == v4 )
            break;
          while ( *(_DWORD *)v5 > 0xFFFFFFFD )
          {
            v5 += 80;
            if ( v4 == v5 )
              goto LABEL_19;
          }
          if ( v4 == v5 )
            break;
          v6 = *(_QWORD **)(v5 + 24);
          v7 = *(_QWORD **)(v5 + 16);
          if ( v6 != v7 )
            goto LABEL_9;
LABEL_53:
          v8 = &v7[*(unsigned int *)(v5 + 36)];
          if ( v7 == v8 )
          {
            v70 = v7;
          }
          else
          {
            do
            {
              if ( a2 == *v7 )
                break;
              ++v7;
            }
            while ( v8 != v7 );
            v70 = v8;
          }
        }
LABEL_19:
        v11 = v72;
        v12 = &v72[(unsigned int)v73];
        if ( v72 != v12 )
        {
          v13 = v72;
          while ( 1 )
          {
            v14 = *(_DWORD *)(a1 + 96);
            v15 = *v13;
            if ( v14 )
            {
              v16 = *(_QWORD *)(a1 + 80);
              v17 = (v14 - 1) & (37 * v15);
              v18 = (int *)(v16 + 80LL * v17);
              v19 = *v18;
              if ( v15 == *v18 )
              {
LABEL_23:
                v20 = (_QWORD *)*((_QWORD *)v18 + 3);
                if ( v20 == *((_QWORD **)v18 + 2) )
                  v21 = (unsigned int)v18[9];
                else
                  v21 = (unsigned int)v18[8];
                v22 = &v20[v21];
                if ( v20 != v22 )
                {
                  while ( 1 )
                  {
                    v23 = *v20;
                    v24 = v20;
                    if ( *v20 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v22 == ++v20 )
                      goto LABEL_28;
                  }
                  while ( v22 != v24 )
                  {
                    if ( *(_BYTE *)(v23 + 16) == 77 )
                    {
                      v42 = *(_DWORD *)(a1 + 32);
                      if ( v42 )
                      {
                        v43 = v42 - 1;
                        v44 = *(_QWORD *)(a1 + 16);
                        v45 = v43 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
                        v46 = (__int64 *)(v44 + 16LL * v45);
                        v47 = *v46;
                        if ( v23 == *v46 )
                        {
LABEL_50:
                          *v46 = -16;
                          --*(_DWORD *)(a1 + 24);
                          ++*(_DWORD *)(a1 + 28);
                        }
                        else
                        {
                          v50 = 1;
                          while ( v47 != -8 )
                          {
                            v51 = v50 + 1;
                            v45 = v43 & (v50 + v45);
                            v46 = (__int64 *)(v44 + 16LL * v45);
                            v47 = *v46;
                            if ( *v46 == v23 )
                              goto LABEL_50;
                            v50 = v51;
                          }
                        }
                      }
                    }
                    v41 = v24 + 1;
                    if ( v24 + 1 == v22 )
                      break;
                    while ( 1 )
                    {
                      v23 = *v41;
                      v24 = v41;
                      if ( *v41 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v22 == ++v41 )
                        goto LABEL_28;
                    }
                  }
                }
                goto LABEL_28;
              }
              v52 = 1;
              v53 = 0;
              while ( v19 != -1 )
              {
                if ( v19 == -2 && !v53 )
                  v53 = v18;
                v17 = (v14 - 1) & (v52 + v17);
                v18 = (int *)(v16 + 80LL * v17);
                v19 = *v18;
                if ( v15 == *v18 )
                  goto LABEL_23;
                ++v52;
              }
              v54 = *(_DWORD *)(a1 + 88);
              if ( !v53 )
                v53 = v18;
              ++*(_QWORD *)(a1 + 72);
              v55 = v54 + 1;
              if ( 4 * (v54 + 1) < 3 * v14 )
              {
                if ( v14 - *(_DWORD *)(a1 + 92) - v55 <= v14 >> 3 )
                {
                  sub_143DB20(v71, v14);
                  v63 = *(_DWORD *)(a1 + 96);
                  if ( !v63 )
                  {
LABEL_118:
                    ++*(_DWORD *)(a1 + 88);
                    BUG();
                  }
                  v64 = v63 - 1;
                  v65 = *(_QWORD *)(a1 + 80);
                  v66 = 0;
                  LODWORD(v67) = v64 & (37 * v15);
                  v55 = *(_DWORD *)(a1 + 88) + 1;
                  v68 = 1;
                  v53 = (int *)(v65 + 80LL * (unsigned int)v67);
                  v69 = *v53;
                  if ( v15 != *v53 )
                  {
                    while ( v69 != -1 )
                    {
                      if ( v69 == -2 && !v66 )
                        v66 = v53;
                      v67 = v64 & (unsigned int)(v67 + v68);
                      v53 = (int *)(v65 + 80 * v67);
                      v69 = *v53;
                      if ( v15 == *v53 )
                        goto LABEL_84;
                      ++v68;
                    }
                    if ( v66 )
                      v53 = v66;
                  }
                }
                goto LABEL_84;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 72);
            }
            sub_143DB20(v71, 2 * v14);
            v56 = *(_DWORD *)(a1 + 96);
            if ( !v56 )
              goto LABEL_118;
            v57 = v56 - 1;
            v58 = *(_QWORD *)(a1 + 80);
            LODWORD(v59) = v57 & (37 * v15);
            v55 = *(_DWORD *)(a1 + 88) + 1;
            v53 = (int *)(v58 + 80LL * (unsigned int)v59);
            v60 = *v53;
            if ( v15 != *v53 )
            {
              v61 = 1;
              v62 = 0;
              while ( v60 != -1 )
              {
                if ( !v62 && v60 == -2 )
                  v62 = v53;
                v59 = v57 & (unsigned int)(v59 + v61);
                v53 = (int *)(v58 + 80 * v59);
                v60 = *v53;
                if ( v15 == *v53 )
                  goto LABEL_84;
                ++v61;
              }
              if ( v62 )
                v53 = v62;
            }
LABEL_84:
            *(_DWORD *)(a1 + 88) = v55;
            if ( *v53 != -1 )
              --*(_DWORD *)(a1 + 92);
            *v53 = v15;
            *((_QWORD *)v53 + 1) = 0;
            *((_QWORD *)v53 + 2) = v53 + 12;
            *((_QWORD *)v53 + 3) = v53 + 12;
            *((_QWORD *)v53 + 4) = 4;
            v53[10] = 0;
LABEL_28:
            v25 = *(_DWORD *)(a1 + 64);
            if ( v25 )
            {
              v26 = v25 - 1;
              v27 = *(_QWORD *)(a1 + 48);
              v28 = 1;
              v29 = v26 & (37 * v15);
              v30 = (int *)(v27 + 80LL * v29);
              v31 = *v30;
              if ( v15 == *v30 )
              {
LABEL_30:
                v32 = *((_QWORD *)v30 + 3);
                if ( v32 != *((_QWORD *)v30 + 2) )
                  _libc_free(v32);
                *v30 = -2;
                --*(_DWORD *)(a1 + 56);
                ++*(_DWORD *)(a1 + 60);
              }
              else
              {
                while ( v31 != -1 )
                {
                  v29 = v26 & (v28 + v29);
                  v30 = (int *)(v27 + 80LL * v29);
                  v31 = *v30;
                  if ( v15 == *v30 )
                    goto LABEL_30;
                  ++v28;
                }
              }
            }
            v33 = *(_DWORD *)(a1 + 96);
            if ( v33 )
            {
              v34 = v33 - 1;
              v35 = *(_QWORD *)(a1 + 80);
              v36 = 1;
              v37 = (v33 - 1) & (37 * v15);
              v38 = (int *)(v35 + 80LL * v37);
              v39 = *v38;
              if ( v15 == *v38 )
              {
LABEL_35:
                v40 = *((_QWORD *)v38 + 3);
                if ( v40 != *((_QWORD *)v38 + 2) )
                  _libc_free(v40);
                *v38 = -2;
                --*(_DWORD *)(a1 + 88);
                ++*(_DWORD *)(a1 + 92);
              }
              else
              {
                while ( v39 != -1 )
                {
                  v37 = v34 & (v36 + v37);
                  v38 = (int *)(v35 + 80LL * v37);
                  v39 = *v38;
                  if ( v15 == *v38 )
                    goto LABEL_35;
                  ++v36;
                }
              }
            }
            if ( v12 == ++v13 )
            {
              v11 = v72;
              break;
            }
          }
        }
        if ( v11 != (int *)v74 )
          _libc_free((unsigned __int64)v11);
      }
    }
  }
}
