// Function: sub_248F0C0
// Address: 0x248f0c0
//
__int64 __fastcall sub_248F0C0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64),
        __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  char v7; // al
  __int64 v8; // r12
  const char *v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r14
  const char *v13; // r15
  __int64 v14; // rbx
  unsigned __int8 v15; // al
  unsigned __int8 **v16; // rdx
  __int64 v17; // r12
  unsigned __int8 *v18; // r13
  unsigned __int8 v19; // al
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int128 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int8 v25; // al
  int v26; // ebx
  unsigned __int8 **v27; // rdx
  unsigned __int8 *v28; // rax
  __int64 v29; // r8
  int v30; // ebx
  unsigned int v31; // esi
  __int64 v32; // r9
  int v33; // r11d
  __int64 *v34; // rdx
  unsigned __int64 v35; // r15
  unsigned int v36; // edi
  __int64 *v37; // rax
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  __int64 v41; // r13
  int v42; // esi
  __int64 v43; // rdx
  unsigned __int8 v44; // al
  int v45; // ecx
  unsigned __int64 v46; // r9
  __int64 v47; // rbx
  __int64 *v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rdx
  int v52; // r10d
  int v53; // r10d
  __int64 v54; // rsi
  __int64 v55; // r9
  __int64 v56; // rax
  int v57; // r15d
  __int64 *v58; // r11
  int v59; // r9d
  int v60; // r9d
  __int64 v61; // rsi
  __int64 *v62; // r10
  int v63; // r11d
  __int64 v64; // rax
  char *v65; // r8
  __int64 v66; // r12
  __int64 v67; // rdi
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rsi
  char *v71; // rax
  _QWORD *v72; // rbx
  _QWORD *v74; // rax
  _QWORD *v75; // r14
  __int64 v76; // rdi
  __int64 v77; // r15
  unsigned __int64 v78; // rax
  unsigned int *v79; // r15
  unsigned int *v80; // rdi
  int v81; // esi
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v85; // [rsp+18h] [rbp-98h]
  __int64 v88; // [rsp+30h] [rbp-80h]
  __int64 v89; // [rsp+38h] [rbp-78h]
  __int64 v90; // [rsp+40h] [rbp-70h]
  char v91; // [rsp+48h] [rbp-68h]
  unsigned int v92; // [rsp+48h] [rbp-68h]
  unsigned int v93; // [rsp+48h] [rbp-68h]
  char v94; // [rsp+4Fh] [rbp-61h]
  __int64 v95; // [rsp+50h] [rbp-60h]
  __int64 v97; // [rsp+60h] [rbp-50h]
  __int64 v98; // [rsp+68h] [rbp-48h]
  const char *v99; // [rsp+70h] [rbp-40h]
  __int64 v100; // [rsp+78h] [rbp-38h]
  __int64 v101; // [rsp+78h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v88 = *(_QWORD *)(a2 + 32);
  v85 = a2 + 24;
  if ( v88 == a2 + 24 )
    return a1;
  do
  {
    v5 = v88 - 56;
    if ( !v88 )
      v5 = 0;
    if ( !sub_B2FC80(v5) )
    {
      v90 = v5 + 72;
      v95 = *(_QWORD *)(v5 + 80);
      if ( v95 != v5 + 72 )
      {
        while ( 1 )
        {
          if ( !v95 )
            BUG();
          v100 = v95 + 24;
          v6 = *(_QWORD *)(v95 + 32);
          if ( v6 != v95 + 24 )
            break;
LABEL_45:
          v95 = *(_QWORD *)(v95 + 8);
          if ( v90 == v95 )
            goto LABEL_3;
        }
LABEL_14:
        while ( 2 )
        {
          if ( !v6 )
            BUG();
          v7 = *(_BYTE *)(v6 - 24);
          if ( v7 == 85 )
          {
            v8 = *(_QWORD *)(v6 - 56);
            if ( !v8
              || !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v6 + 56) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
            {
              goto LABEL_13;
            }
          }
          else
          {
            if ( v7 != 34 && v7 != 40 )
              goto LABEL_13;
            v8 = *(_QWORD *)(v6 - 56);
          }
          if ( v8 )
          {
            if ( !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v6 + 56) && (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
            {
              v9 = sub_BD5D20(v8);
              v11 = v10;
              v94 = sub_2484E50(v8, *a3);
              v12 = sub_B10CD0(v6 + 24);
              if ( v12 )
              {
                v91 = 1;
                v89 = v6;
                v13 = v9;
                v14 = v11;
                while ( 1 )
                {
                  v97 = v12 - 16;
                  v15 = *(_BYTE *)(v12 - 16);
                  v16 = (v15 & 2) != 0
                      ? *(unsigned __int8 ***)(v12 - 32)
                      : (unsigned __int8 **)(v97 - 8LL * ((v15 >> 2) & 0xF));
                  v17 = 0;
                  v18 = sub_AF34D0(*v16);
                  v99 = byte_3F871B3;
                  if ( v18 )
                    break;
LABEL_31:
                  *(_QWORD *)&v22 = v99;
                  *((_QWORD *)&v22 + 1) = v17;
                  v23 = sub_C16C50(v22);
                  *(_QWORD *)&v22 = v13;
                  *((_QWORD *)&v22 + 1) = v14;
                  v24 = v23;
                  v98 = sub_C16C50(v22);
                  if ( v94 )
                  {
                    if ( v91 || !a4(a5, v98) )
                      v98 = 0;
                    else
                      v94 = 0;
                  }
                  v25 = *(_BYTE *)(v12 - 16);
                  v26 = *(_DWORD *)(v12 + 4);
                  if ( (v25 & 2) != 0 )
                    v27 = *(unsigned __int8 ***)(v12 - 32);
                  else
                    v27 = (unsigned __int8 **)(v97 - 8LL * ((v25 >> 2) & 0xF));
                  v28 = sub_AF34D0(*v27);
                  v29 = *(unsigned __int16 *)(v12 + 2);
                  v30 = v26 - *((_DWORD *)v28 + 4);
                  v31 = *(_DWORD *)(a1 + 24);
                  if ( !v31 )
                  {
                    ++*(_QWORD *)a1;
                    goto LABEL_82;
                  }
                  v32 = *(_QWORD *)(a1 + 8);
                  v33 = 1;
                  v34 = 0;
                  v35 = ((0xBF58476D1CE4E5B9LL * v24) >> 31) ^ (0xBF58476D1CE4E5B9LL * v24);
                  v36 = v35 & (v31 - 1);
                  v37 = (__int64 *)(v32 + 24LL * v36);
                  v38 = *v37;
                  if ( v24 != *v37 )
                  {
                    while ( v38 != -1 )
                    {
                      if ( v38 == -2 && !v34 )
                        v34 = v37;
                      v36 = (v31 - 1) & (v33 + v36);
                      v37 = (__int64 *)(v32 + 24LL * v36);
                      v38 = *v37;
                      if ( v24 == *v37 )
                        goto LABEL_39;
                      ++v33;
                    }
                    if ( !v34 )
                      v34 = v37;
                    ++*(_QWORD *)a1;
                    v45 = *(_DWORD *)(a1 + 16) + 1;
                    if ( 4 * v45 >= 3 * v31 )
                    {
LABEL_82:
                      v92 = v29;
                      sub_EE0D70(a1, 2 * v31);
                      v52 = *(_DWORD *)(a1 + 24);
                      if ( !v52 )
                        goto LABEL_147;
                      v53 = v52 - 1;
                      v54 = *(_QWORD *)(a1 + 8);
                      v29 = v92;
                      v45 = *(_DWORD *)(a1 + 16) + 1;
                      LODWORD(v55) = v53 & (((0xBF58476D1CE4E5B9LL * v24) >> 31) ^ (484763065 * v24));
                      v34 = (__int64 *)(v54 + 24LL * (unsigned int)v55);
                      v56 = *v34;
                      if ( v24 != *v34 )
                      {
                        v57 = 1;
                        v58 = 0;
                        while ( v56 != -1 )
                        {
                          if ( v56 == -2 && !v58 )
                            v58 = v34;
                          v55 = v53 & (unsigned int)(v55 + v57);
                          v34 = (__int64 *)(v54 + 24 * v55);
                          v56 = *v34;
                          if ( v24 == *v34 )
                            goto LABEL_57;
                          ++v57;
                        }
                        if ( v58 )
                          v34 = v58;
                      }
                    }
                    else if ( v31 - *(_DWORD *)(a1 + 20) - v45 <= v31 >> 3 )
                    {
                      v93 = v29;
                      sub_EE0D70(a1, v31);
                      v59 = *(_DWORD *)(a1 + 24);
                      if ( !v59 )
                      {
LABEL_147:
                        ++*(_DWORD *)(a1 + 16);
                        BUG();
                      }
                      v60 = v59 - 1;
                      v61 = *(_QWORD *)(a1 + 8);
                      v62 = 0;
                      LODWORD(v35) = v60 & v35;
                      v29 = v93;
                      v63 = 1;
                      v45 = *(_DWORD *)(a1 + 16) + 1;
                      v34 = (__int64 *)(v61 + 24LL * (unsigned int)v35);
                      v64 = *v34;
                      if ( v24 != *v34 )
                      {
                        while ( v64 != -1 )
                        {
                          if ( v64 == -2 && !v62 )
                            v62 = v34;
                          v35 = v60 & (unsigned int)(v35 + v63);
                          v34 = (__int64 *)(v61 + 24 * v35);
                          v64 = *v34;
                          if ( v24 == *v34 )
                            goto LABEL_57;
                          ++v63;
                        }
                        if ( v62 )
                          v34 = v62;
                      }
                    }
LABEL_57:
                    *(_DWORD *)(a1 + 16) = v45;
                    if ( *v34 != -1 )
                      --*(_DWORD *)(a1 + 20);
                    *v34 = v24;
                    v46 = 1;
                    v34[1] = (__int64)(v34 + 3);
                    v41 = (__int64)(v34 + 1);
                    v34[2] = 0;
                    v47 = (v29 << 32) | (unsigned __int16)v30;
LABEL_60:
                    sub_C8D5F0(v41, (const void *)(v41 + 16), v46, 0x10u, v29, v46);
                    v39 = *(unsigned int *)(v41 + 8);
LABEL_61:
                    v48 = (__int64 *)(*(_QWORD *)v41 + 16 * v39);
                    *v48 = v47;
                    v13 = v99;
                    v14 = v17;
                    v48[1] = v98;
                    ++*(_DWORD *)(v41 + 8);
                    v44 = *(_BYTE *)(v12 - 16);
                    if ( (v44 & 2) != 0 )
                      goto LABEL_43;
                    goto LABEL_62;
                  }
LABEL_39:
                  v39 = *((unsigned int *)v37 + 4);
                  v40 = *((unsigned int *)v37 + 5);
                  v41 = (__int64)(v37 + 1);
                  v42 = *((_DWORD *)v37 + 4);
                  if ( v39 >= v40 )
                  {
                    v46 = v39 + 1;
                    v47 = (v29 << 32) | (unsigned __int16)v30;
                    if ( v39 + 1 > v40 )
                      goto LABEL_60;
                    goto LABEL_61;
                  }
                  v43 = v37[1] + 16 * v39;
                  if ( v43 )
                  {
                    *(_DWORD *)v43 = (unsigned __int16)v30;
                    *(_DWORD *)(v43 + 4) = v29;
                    *(_QWORD *)(v43 + 8) = v98;
                    v42 = *((_DWORD *)v37 + 4);
                  }
                  v13 = v99;
                  v14 = v17;
                  *((_DWORD *)v37 + 4) = v42 + 1;
                  v44 = *(_BYTE *)(v12 - 16);
                  if ( (v44 & 2) != 0 )
                  {
LABEL_43:
                    if ( *(_DWORD *)(v12 - 24) != 2 )
                      goto LABEL_44;
                    v49 = *(_QWORD *)(v12 - 32);
                    goto LABEL_64;
                  }
LABEL_62:
                  if ( ((*(_WORD *)(v12 - 16) >> 6) & 0xF) != 2 )
                    goto LABEL_44;
                  v49 = v97 - 8LL * ((v44 >> 2) & 0xF);
LABEL_64:
                  v12 = *(_QWORD *)(v49 + 8);
                  if ( !v12 )
                  {
LABEL_44:
                    v6 = *(_QWORD *)(v89 + 8);
                    if ( v100 == v6 )
                      goto LABEL_45;
                    goto LABEL_14;
                  }
                  v91 = 0;
                }
                v19 = *(v18 - 16);
                if ( (v19 & 2) != 0 )
                {
                  v20 = *(_QWORD *)(*((_QWORD *)v18 - 4) + 24LL);
                  if ( !v20 )
                    goto LABEL_77;
                }
                else
                {
                  v20 = *(_QWORD *)&v18[-8 * ((v19 >> 2) & 0xF) + 8];
                  if ( !v20 )
                    goto LABEL_72;
                }
                v99 = (const char *)sub_B91420(v20);
                v17 = v21;
                if ( v21 )
                  goto LABEL_31;
                v19 = *(v18 - 16);
                if ( (v19 & 2) != 0 )
                {
LABEL_77:
                  v50 = *(_QWORD *)(*((_QWORD *)v18 - 4) + 16LL);
                  v99 = (const char *)v50;
                  if ( v50 )
                  {
LABEL_73:
                    v99 = (const char *)sub_B91420(v50);
                    v17 = v51;
                    goto LABEL_31;
                  }
                  goto LABEL_78;
                }
LABEL_72:
                v50 = *(_QWORD *)&v18[-8 * ((v19 >> 2) & 0xF)];
                v99 = (const char *)v50;
                if ( v50 )
                  goto LABEL_73;
LABEL_78:
                v17 = 0;
                goto LABEL_31;
              }
            }
          }
LABEL_13:
          v6 = *(_QWORD *)(v6 + 8);
          if ( v100 == v6 )
            goto LABEL_45;
          continue;
        }
      }
    }
LABEL_3:
    v88 = *(_QWORD *)(v88 + 8);
  }
  while ( v85 != v88 );
  if ( *(_DWORD *)(a1 + 16) )
  {
    v74 = *(_QWORD **)(a1 + 8);
    v75 = &v74[3 * *(unsigned int *)(a1 + 24)];
    if ( v74 != v75 )
    {
      v72 = *(_QWORD **)(a1 + 8);
      if ( *v74 <= 0xFFFFFFFFFFFFFFFDLL )
      {
LABEL_110:
        if ( v75 != v74 )
        {
          do
          {
            v76 = v72[1];
            v77 = 16LL * *((unsigned int *)v72 + 4);
            v66 = v76 + v77;
            v65 = (char *)(v76 + v77);
            if ( v76 != v76 + v77 )
            {
              v101 = v72[1];
              _BitScanReverse64(&v78, v77 >> 4);
              sub_248ED70(v76, (unsigned __int64 *)(v76 + v77), 2LL * (int)(63 - (v78 ^ 0x3F)));
              if ( (unsigned __int64)v77 <= 0x100 )
              {
                sub_2485E10(v101, v66);
              }
              else
              {
                v79 = (unsigned int *)(v101 + 256);
                sub_2485E10(v101, v101 + 256);
                if ( v66 != v101 + 256 )
                {
                  do
                  {
                    v80 = v79;
                    v79 += 4;
                    sub_2485DB0(v80);
                  }
                  while ( (unsigned int *)v66 != v79 );
                }
              }
              v65 = (char *)v72[1];
              v71 = &v65[16 * *((unsigned int *)v72 + 4)];
              if ( v65 == v71 )
              {
                v66 = v72[1];
              }
              else
              {
                v66 = (__int64)(v65 + 16);
                if ( v71 != v65 + 16 )
                {
                  while ( 1 )
                  {
                    v81 = *(_DWORD *)(v66 - 16);
                    v82 = v66 - 16;
                    if ( v81 == *(_DWORD *)v66
                      && *(_DWORD *)(v66 - 12) == *(_DWORD *)(v66 + 4)
                      && *(_QWORD *)(v66 - 8) == *(_QWORD *)(v66 + 8) )
                    {
                      break;
                    }
                    v66 += 16;
                    if ( v71 == (char *)v66 )
                      goto LABEL_101;
                  }
                  if ( v71 == (char *)v82 )
                  {
                    v66 = (__int64)&v65[16 * *((unsigned int *)v72 + 4)];
                  }
                  else
                  {
                    v83 = v66 + 16;
                    if ( v71 != (char *)(v66 + 16) )
                    {
                      while ( 1 )
                      {
                        if ( *(_DWORD *)v83 != v81
                          || *(_DWORD *)(v82 + 4) != *(_DWORD *)(v83 + 4)
                          || *(_QWORD *)(v82 + 8) != *(_QWORD *)(v83 + 8) )
                        {
                          v82 += 16;
                          *(_QWORD *)v82 = *(_QWORD *)v83;
                          *(_QWORD *)(v82 + 8) = *(_QWORD *)(v83 + 8);
                        }
                        v83 += 16;
                        if ( v71 == (char *)v83 )
                          break;
                        v81 = *(_DWORD *)v82;
                      }
                      v65 = (char *)v72[1];
                      v66 = v82 + 16;
                      v67 = &v65[16 * *((unsigned int *)v72 + 4)] - v71;
                      v68 = v67 >> 4;
                      if ( v67 > 0 )
                      {
                        v69 = v82 + 16;
                        do
                        {
                          v70 = *(_QWORD *)v71;
                          v69 += 16;
                          v71 += 16;
                          *(_QWORD *)(v69 - 16) = v70;
                          *(_QWORD *)(v69 - 8) = *((_QWORD *)v71 - 1);
                          --v68;
                        }
                        while ( v68 );
                        v65 = (char *)v72[1];
                        v66 += v67;
                      }
                    }
                  }
                }
              }
            }
LABEL_101:
            v72 += 3;
            *((_DWORD *)v72 - 2) = (v66 - (__int64)v65) >> 4;
            if ( v72 == v75 )
              break;
            while ( *v72 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v72 += 3;
              if ( v75 == v72 )
                return a1;
            }
          }
          while ( v75 != v72 );
        }
      }
      else
      {
        while ( 1 )
        {
          v74 += 3;
          if ( v75 == v74 )
            break;
          v72 = v74;
          if ( *v74 <= 0xFFFFFFFFFFFFFFFDLL )
            goto LABEL_110;
        }
      }
    }
  }
  return a1;
}
