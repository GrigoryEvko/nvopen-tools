// Function: sub_3589A20
// Address: 0x3589a20
//
unsigned __int64 __fastcall sub_3589A20(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // r13
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // rdi
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 *v13; // r12
  __int64 *v14; // r9
  unsigned int v15; // esi
  __int64 v16; // r15
  __int64 v17; // r8
  int v18; // r11d
  unsigned __int64 v19; // rbx
  unsigned int i; // edi
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // r12
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // r9
  int v27; // r11d
  unsigned __int64 v28; // rbx
  __int64 *v29; // rdx
  unsigned int k; // r8d
  __int64 *v31; // rcx
  __int64 v32; // rdi
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  int v35; // ebx
  int v36; // esi
  int v37; // esi
  __int64 v38; // r8
  __int64 *v39; // r9
  int v40; // r10d
  unsigned int n; // ecx
  __int64 v42; // rdi
  unsigned int v43; // ecx
  int v44; // esi
  int v45; // esi
  __int64 v46; // rcx
  int v47; // r10d
  unsigned __int64 v48; // r8
  unsigned int v49; // edx
  __int64 v50; // rdi
  unsigned int v51; // edx
  unsigned int v52; // edi
  int v53; // ecx
  unsigned int v54; // r8d
  int v55; // eax
  int v56; // ecx
  int v57; // eax
  int v58; // esi
  __int64 v59; // r8
  unsigned int v60; // eax
  __int64 v61; // rdi
  int v62; // r10d
  __int64 *v63; // r9
  int v64; // ecx
  int v65; // eax
  int v66; // edx
  __int64 v67; // rsi
  unsigned __int64 v68; // rdi
  unsigned int v69; // ebx
  int j; // r8d
  __int64 v71; // rcx
  unsigned int v72; // ebx
  int v73; // eax
  int v74; // eax
  __int64 v75; // rdi
  __int64 *v76; // r8
  unsigned int v77; // ebx
  int v78; // r9d
  __int64 v79; // rsi
  int v80; // ecx
  int v81; // ecx
  int v82; // ecx
  int v83; // ecx
  __int64 v84; // rdi
  __int64 *v85; // r8
  unsigned int v86; // ebx
  int m; // r9d
  __int64 v88; // rsi
  unsigned int v89; // ebx
  __int64 v90; // [rsp+0h] [rbp-80h]
  __int64 v91; // [rsp+0h] [rbp-80h]
  __int64 v92; // [rsp+8h] [rbp-78h]
  unsigned int v93; // [rsp+14h] [rbp-6Ch]
  __int64 v94; // [rsp+18h] [rbp-68h]
  __int64 v95; // [rsp+20h] [rbp-60h]
  __int64 *v96; // [rsp+28h] [rbp-58h]
  __int64 *v97; // [rsp+28h] [rbp-58h]
  __int64 *v98; // [rsp+28h] [rbp-58h]
  unsigned __int64 v99; // [rsp+30h] [rbp-50h]
  unsigned __int64 v100; // [rsp+38h] [rbp-48h]
  int v101; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v102[7]; // [rsp+48h] [rbp-38h] BYREF

  result = a2 + 320;
  v3 = *(_QWORD *)(a2 + 328);
  v95 = a2 + 320;
  v92 = a1 + 968;
  if ( v3 != a2 + 320 )
  {
LABEL_4:
    if ( *(_DWORD *)(v3 + 120) <= 1u )
      goto LABEL_3;
    v5 = *(_DWORD *)(a1 + 992);
    if ( v5 )
    {
      v6 = 1;
      v7 = *(_QWORD *)(a1 + 976);
      v8 = 0;
      v9 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == v3 )
      {
LABEL_7:
        v12 = v10[1];
LABEL_8:
        v102[0] = v12;
        sub_3588500(a1 + 40, v102);
        v13 = *(__int64 **)(v3 + 112);
        result = *(unsigned int *)(v3 + 120);
        v14 = &v13[result];
        if ( v13 == v14 )
          goto LABEL_3;
        v99 = 0;
        v100 = (unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32;
        v94 = a1 + 72;
        while ( 1 )
        {
          v15 = *(_DWORD *)(a1 + 96);
          v16 = *v13;
          if ( !v15 )
          {
            ++*(_QWORD *)(a1 + 72);
            goto LABEL_52;
          }
          v17 = *(_QWORD *)(a1 + 80);
          v18 = 1;
          v19 = ((0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4))) >> 31)
              ^ (0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)));
          result = 0;
          for ( i = v19 & (v15 - 1); ; i = (v15 - 1) & v52 )
          {
            v21 = (__int64 *)(v17 + 24LL * i);
            v22 = *v21;
            if ( v3 == *v21 && v16 == v21[1] )
            {
              v99 += v21[2];
              goto LABEL_21;
            }
            if ( v22 == -4096 )
              break;
            if ( v22 == -8192 && v21[1] == -8192 && !result )
              result = v17 + 24LL * i;
LABEL_63:
            v52 = v18 + i;
            ++v18;
          }
          if ( v21[1] != -4096 )
            goto LABEL_63;
          v64 = *(_DWORD *)(a1 + 88);
          if ( !result )
            result = v17 + 24LL * i;
          ++*(_QWORD *)(a1 + 72);
          v53 = v64 + 1;
          if ( 4 * v53 < 3 * v15 )
          {
            if ( v15 - *(_DWORD *)(a1 + 92) - v53 > v15 >> 3 )
              goto LABEL_66;
            v98 = v14;
            sub_35895B0(v94, v15);
            v65 = *(_DWORD *)(a1 + 96);
            if ( !v65 )
            {
LABEL_165:
              ++*(_DWORD *)(a1 + 88);
              BUG();
            }
            v66 = v65 - 1;
            v67 = *(_QWORD *)(a1 + 80);
            v14 = v98;
            v68 = 0;
            v69 = (v65 - 1) & v19;
            for ( j = 1; ; ++j )
            {
              result = v67 + 24LL * v69;
              v71 = *(_QWORD *)result;
              if ( v3 == *(_QWORD *)result && v16 == *(_QWORD *)(result + 8) )
                goto LABEL_65;
              if ( v71 == -4096 )
              {
                if ( *(_QWORD *)(result + 8) == -4096 )
                {
                  v53 = *(_DWORD *)(a1 + 88) + 1;
                  if ( v68 )
                    result = v68;
                  goto LABEL_66;
                }
              }
              else if ( v71 == -8192 && *(_QWORD *)(result + 8) == -8192 && !v68 )
              {
                v68 = v67 + 24LL * v69;
              }
              v72 = j + v69;
              v69 = v66 & v72;
            }
          }
LABEL_52:
          v97 = v14;
          sub_35895B0(v94, 2 * v15);
          v44 = *(_DWORD *)(a1 + 96);
          if ( !v44 )
            goto LABEL_165;
          v45 = v44 - 1;
          v14 = v97;
          v47 = 1;
          v48 = 0;
          v49 = v45
              & (((0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4))) >> 31)
               ^ (484763065 * (v100 | ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4))));
          while ( 2 )
          {
            v46 = *(_QWORD *)(a1 + 80);
            result = v46 + 24LL * v49;
            v50 = *(_QWORD *)result;
            if ( v3 == *(_QWORD *)result && v16 == *(_QWORD *)(result + 8) )
            {
LABEL_65:
              v53 = *(_DWORD *)(a1 + 88) + 1;
              goto LABEL_66;
            }
            if ( v50 != -4096 )
            {
              if ( v50 == -8192 && *(_QWORD *)(result + 8) == -8192 && !v48 )
                v48 = v46 + 24LL * v49;
              goto LABEL_60;
            }
            if ( *(_QWORD *)(result + 8) != -4096 )
            {
LABEL_60:
              v51 = v47 + v49;
              ++v47;
              v49 = v45 & v51;
              continue;
            }
            break;
          }
          v53 = *(_DWORD *)(a1 + 88) + 1;
          if ( v48 )
            result = v48;
LABEL_66:
          *(_DWORD *)(a1 + 88) = v53;
          if ( *(_QWORD *)result != -4096 || *(_QWORD *)(result + 8) != -4096 )
            --*(_DWORD *)(a1 + 92);
          *(_QWORD *)result = v3;
          *(_QWORD *)(result + 8) = v16;
          *(_QWORD *)(result + 16) = 0;
LABEL_21:
          if ( v14 == ++v13 )
          {
            if ( v99 )
            {
              v93 = 1;
              if ( v99 > 0xFFFFFFFF )
              {
                v93 = v99 / 0xFFFFFFFF + 1;
                v99 /= v93;
              }
              v23 = *(__int64 **)(v3 + 112);
              result = (unsigned __int64)&v23[*(unsigned int *)(v3 + 120)];
              v96 = (__int64 *)result;
              if ( (__int64 *)result != v23 )
              {
                v24 = *(_DWORD *)(a1 + 96);
                v25 = *v23;
                if ( v24 )
                {
LABEL_27:
                  v26 = *(_QWORD *)(a1 + 80);
                  v27 = 1;
                  v28 = ((0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4))) >> 31)
                      ^ (0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)));
                  v29 = 0;
                  for ( k = v28 & (v24 - 1); ; k = (v24 - 1) & v54 )
                  {
                    v31 = (__int64 *)(v26 + 24LL * k);
                    v32 = *v31;
                    if ( v3 == *v31 && v25 == v31[1] )
                    {
                      v33 = v31[2] / (unsigned __int64)v93;
                      goto LABEL_37;
                    }
                    if ( v32 == -4096 )
                    {
                      if ( v31[1] == -4096 )
                      {
                        if ( !v29 )
                          v29 = (__int64 *)(v26 + 24LL * k);
                        v81 = *(_DWORD *)(a1 + 88);
                        ++*(_QWORD *)(a1 + 72);
                        v80 = v81 + 1;
                        if ( 4 * v80 < 3 * v24 )
                        {
                          if ( v24 - *(_DWORD *)(a1 + 92) - v80 <= v24 >> 3 )
                          {
                            v91 = v25;
                            sub_35895B0(v94, v24);
                            v82 = *(_DWORD *)(a1 + 96);
                            if ( !v82 )
                            {
LABEL_164:
                              ++*(_DWORD *)(a1 + 88);
                              BUG();
                            }
                            v83 = v82 - 1;
                            v25 = v91;
                            v85 = 0;
                            v86 = v83 & v28;
                            for ( m = 1; ; ++m )
                            {
                              v84 = *(_QWORD *)(a1 + 80);
                              v29 = (__int64 *)(v84 + 24LL * v86);
                              v88 = *v29;
                              if ( v3 == *v29 && v91 == v29[1] )
                                goto LABEL_116;
                              if ( v88 == -4096 )
                              {
                                if ( v29[1] == -4096 )
                                {
                                  v80 = *(_DWORD *)(a1 + 88) + 1;
                                  if ( v85 )
                                    v29 = v85;
                                  goto LABEL_117;
                                }
                              }
                              else if ( v88 == -8192 && v29[1] == -8192 && !v85 )
                              {
                                v85 = (__int64 *)(v84 + 24LL * v86);
                              }
                              v89 = m + v86;
                              v86 = v83 & v89;
                            }
                          }
                          goto LABEL_117;
                        }
                        goto LABEL_42;
                      }
                    }
                    else if ( v32 == -8192 && v31[1] == -8192 && !v29 )
                    {
                      v29 = (__int64 *)(v26 + 24LL * k);
                    }
                    v54 = v27 + k;
                    ++v27;
                  }
                }
                while ( 1 )
                {
                  ++*(_QWORD *)(a1 + 72);
LABEL_42:
                  v90 = v25;
                  sub_35895B0(v94, 2 * v24);
                  v36 = *(_DWORD *)(a1 + 96);
                  if ( !v36 )
                    goto LABEL_164;
                  v25 = v90;
                  v37 = v36 - 1;
                  v38 = *(_QWORD *)(a1 + 80);
                  v39 = 0;
                  v40 = 1;
                  for ( n = v37
                          & (((0xBF58476D1CE4E5B9LL * (v100 | ((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4))) >> 31)
                           ^ (484763065 * (v100 | ((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4)))); ; n = v37 & v43 )
                  {
                    v29 = (__int64 *)(v38 + 24LL * n);
                    v42 = *v29;
                    if ( v3 == *v29 && v90 == v29[1] )
                      break;
                    if ( v42 == -4096 )
                    {
                      if ( v29[1] == -4096 )
                      {
                        v80 = *(_DWORD *)(a1 + 88) + 1;
                        if ( v39 )
                          v29 = v39;
                        goto LABEL_117;
                      }
                    }
                    else if ( v42 == -8192 && v29[1] == -8192 && !v39 )
                    {
                      v39 = (__int64 *)(v38 + 24LL * n);
                    }
                    v43 = v40 + n;
                    ++v40;
                  }
LABEL_116:
                  v80 = *(_DWORD *)(a1 + 88) + 1;
LABEL_117:
                  *(_DWORD *)(a1 + 88) = v80;
                  if ( *v29 != -4096 || v29[1] != -4096 )
                    --*(_DWORD *)(a1 + 92);
                  *v29 = v3;
                  LODWORD(v33) = 0;
                  v29[1] = v25;
                  v29[2] = 0;
LABEL_37:
                  v34 = sub_2E3A070(*(__int64 **)(a1 + 1296));
                  v35 = sub_2E441C0(v34, v3, (__int64)v23);
                  result = sub_F02DB0(&v101, v33, v99);
                  if ( v101 != v35 )
                    result = (unsigned __int64)sub_2E32F90(v3, (__int64)v23, v101);
                  if ( ++v23 == v96 )
                    break;
                  v24 = *(_DWORD *)(a1 + 96);
                  v25 = *v23;
                  if ( v24 )
                    goto LABEL_27;
                }
              }
            }
LABEL_3:
            v3 = *(_QWORD *)(v3 + 8);
            if ( v95 == v3 )
              return result;
            goto LABEL_4;
          }
        }
      }
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v8 )
          v8 = v10;
        v9 = (v5 - 1) & (v6 + v9);
        v10 = (__int64 *)(v7 + 16LL * v9);
        v11 = *v10;
        if ( *v10 == v3 )
          goto LABEL_7;
        ++v6;
      }
      if ( !v8 )
        v8 = v10;
      v55 = *(_DWORD *)(a1 + 984);
      ++*(_QWORD *)(a1 + 968);
      v56 = v55 + 1;
      if ( 4 * (v55 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(a1 + 988) - v56 <= v5 >> 3 )
        {
          sub_35793B0(v92, v5);
          v73 = *(_DWORD *)(a1 + 992);
          if ( !v73 )
          {
LABEL_163:
            ++*(_DWORD *)(a1 + 984);
            BUG();
          }
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a1 + 976);
          v76 = 0;
          v77 = v74 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v78 = 1;
          v56 = *(_DWORD *)(a1 + 984) + 1;
          v8 = (__int64 *)(v75 + 16LL * v77);
          v79 = *v8;
          if ( *v8 != v3 )
          {
            while ( v79 != -4096 )
            {
              if ( !v76 && v79 == -8192 )
                v76 = v8;
              v77 = v74 & (v78 + v77);
              v8 = (__int64 *)(v75 + 16LL * v77);
              v79 = *v8;
              if ( v3 == *v8 )
                goto LABEL_81;
              ++v78;
            }
            if ( v76 )
              v8 = v76;
          }
        }
LABEL_81:
        *(_DWORD *)(a1 + 984) = v56;
        if ( *v8 != -4096 )
          --*(_DWORD *)(a1 + 988);
        *v8 = v3;
        v12 = 0;
        v8[1] = 0;
        goto LABEL_8;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 968);
    }
    sub_35793B0(v92, 2 * v5);
    v57 = *(_DWORD *)(a1 + 992);
    if ( !v57 )
      goto LABEL_163;
    v58 = v57 - 1;
    v59 = *(_QWORD *)(a1 + 976);
    v60 = (v57 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v56 = *(_DWORD *)(a1 + 984) + 1;
    v8 = (__int64 *)(v59 + 16LL * v60);
    v61 = *v8;
    if ( *v8 != v3 )
    {
      v62 = 1;
      v63 = 0;
      while ( v61 != -4096 )
      {
        if ( !v63 && v61 == -8192 )
          v63 = v8;
        v60 = v58 & (v62 + v60);
        v8 = (__int64 *)(v59 + 16LL * v60);
        v61 = *v8;
        if ( *v8 == v3 )
          goto LABEL_81;
        ++v62;
      }
      if ( v63 )
        v8 = v63;
    }
    goto LABEL_81;
  }
  return result;
}
