// Function: sub_18AC760
// Address: 0x18ac760
//
void __fastcall sub_18AC760(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v4; // r9d
  __int64 v5; // rdi
  int v6; // r10d
  unsigned int v7; // ecx
  __int64 v8; // r13
  __int64 *v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r11
  __int64 v12; // rax
  unsigned int v13; // r8d
  int v14; // esi
  int v15; // esi
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // r9
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 *v26; // r8
  int v27; // ecx
  int v28; // r15d
  int v29; // edi
  int v30; // r11d
  __int64 *v31; // r10
  int v32; // edi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  int v36; // r8d
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rdi
  unsigned int v43; // eax
  __int64 *v44; // r15
  __int64 v45; // rbx
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rbx
  unsigned __int64 v50; // r15
  __int64 *v51; // rcx
  __int64 v52; // rax
  __int64 *v53; // r13
  __int64 v54; // r12
  __int64 *v55; // r15
  __int64 v56; // r9
  int v57; // edx
  unsigned int v58; // esi
  __int64 *v59; // rax
  __int64 v60; // r10
  __int64 v61; // r10
  __int64 v62; // rsi
  unsigned int v63; // edi
  __int64 *v64; // rax
  __int64 v65; // r11
  bool v66; // al
  unsigned int v67; // r8d
  __int64 v68; // rdi
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // rcx
  unsigned __int64 v72; // rax
  bool v73; // r8
  __int64 v74; // rax
  int v75; // edx
  __int64 v76; // rax
  int v77; // eax
  int v78; // eax
  int v79; // edx
  __int64 v80; // r12
  int v81; // ebx
  __int64 *v82; // r10
  int v83; // ebx
  int v84; // ecx
  int v85; // ebx
  int v86; // edi
  __int64 *v87; // r11
  int v88; // r10d
  __int64 v90; // [rsp+10h] [rbp-120h]
  __int64 v91; // [rsp+18h] [rbp-118h]
  __int64 v92; // [rsp+28h] [rbp-108h]
  __int64 v93; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v94; // [rsp+38h] [rbp-F8h]
  __int64 v95; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v96; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v97; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v98; // [rsp+58h] [rbp-D8h] BYREF
  __int64 *v99; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v100; // [rsp+68h] [rbp-C8h]
  _BYTE v101[64]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 *v102; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v103; // [rsp+B8h] [rbp-78h]
  _QWORD v104[14]; // [rsp+C0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v99 = (__int64 *)v101;
  v100 = 0x800000000LL;
  v92 = a2 + 72;
  if ( v2 == a2 + 72 )
    return;
  v91 = a1 + 936;
  do
  {
    v12 = v2 - 24;
    v13 = *(_DWORD *)(a1 + 960);
    if ( !v2 )
      v12 = 0;
    v95 = v12;
    if ( !v13 )
    {
      ++*(_QWORD *)(a1 + 936);
LABEL_9:
      v14 = 2 * v13;
      goto LABEL_10;
    }
    v4 = v13 - 1;
    v5 = *(_QWORD *)(a1 + 944);
    v6 = 1;
    v7 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    LODWORD(v8) = v7;
    v9 = (__int64 *)(v5 + 16LL * v7);
    v10 = *v9;
    v11 = *v9;
    if ( v12 == *v9 )
      goto LABEL_4;
    while ( v11 != -8 )
    {
      v8 = v4 & ((_DWORD)v8 + v6);
      v11 = *(_QWORD *)(v5 + 16 * v8);
      if ( v12 == v11 )
        goto LABEL_4;
      ++v6;
    }
    v30 = 1;
    v31 = 0;
    while ( v10 != -8 )
    {
      if ( !v31 && v10 == -16 )
        v31 = v9;
      v7 = v4 & (v30 + v7);
      v9 = (__int64 *)(v5 + 16LL * v7);
      v10 = *v9;
      if ( v12 == *v9 )
        goto LABEL_48;
      ++v30;
    }
    v32 = *(_DWORD *)(a1 + 952);
    if ( v31 )
      v9 = v31;
    ++*(_QWORD *)(a1 + 936);
    v15 = v32 + 1;
    if ( 4 * (v32 + 1) >= 3 * v13 )
      goto LABEL_9;
    if ( v13 - *(_DWORD *)(a1 + 956) - v15 > v13 >> 3 )
      goto LABEL_45;
    v14 = v13;
LABEL_10:
    sub_18AC470(v91, v14);
    sub_18A8880(v91, &v95, &v102);
    v9 = v102;
    v12 = v95;
    v15 = *(_DWORD *)(a1 + 952) + 1;
LABEL_45:
    *(_DWORD *)(a1 + 952) = v15;
    if ( *v9 != -8 )
      --*(_DWORD *)(a1 + 956);
    v9[1] = 0;
    *v9 = v12;
    v10 = v95;
LABEL_48:
    v9[1] = v10;
    v33 = *(_QWORD *)(a1 + 1000);
    LODWORD(v100) = 0;
    v34 = *(unsigned int *)(v33 + 48);
    if ( (_DWORD)v34 )
    {
      v35 = *(_QWORD *)(v33 + 32);
      v36 = v34 - 1;
      v37 = (v34 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v38 = (__int64 *)(v35 + 16LL * v37);
      v39 = *v38;
      if ( v10 == *v38 )
      {
LABEL_50:
        if ( v38 != (__int64 *)(v35 + 16 * v34) )
        {
          v40 = v38[1];
          if ( v40 )
          {
            v93 = v2;
            v41 = 0;
            v102 = v104;
            v42 = v104;
            v104[0] = v40;
            v103 = 0x800000001LL;
            v43 = 1;
            while ( 1 )
            {
              v44 = (__int64 *)v42[v43 - 1];
              LODWORD(v103) = v43 - 1;
              v45 = *v44;
              if ( (unsigned int)v41 >= HIDWORD(v100) )
              {
                sub_16CD150((__int64)&v99, v101, 0, 8, v36, v39);
                v41 = (unsigned int)v100;
              }
              v99[v41] = v45;
              v46 = v44[3];
              v47 = v44[4];
              v48 = (unsigned int)v103;
              LODWORD(v100) = v100 + 1;
              v49 = v47 - v46;
              v50 = (v47 - v46) >> 3;
              if ( v50 > HIDWORD(v103) - (unsigned __int64)(unsigned int)v103 )
              {
                sub_16CD150((__int64)&v102, v104, v50 + (unsigned int)v103, 8, v36, v39);
                v48 = (unsigned int)v103;
              }
              v42 = v102;
              v51 = &v102[v48];
              if ( v49 > 0 )
              {
                v52 = 0;
                do
                {
                  v51[v52] = *(_QWORD *)(v46 + 8 * v52);
                  ++v52;
                }
                while ( (__int64)(v50 - v52) > 0 );
                v42 = v102;
                LODWORD(v48) = v103;
              }
              LODWORD(v103) = v48 + v50;
              v43 = v48 + v50;
              if ( !((_DWORD)v48 + (_DWORD)v50) )
                break;
              v41 = (unsigned int)v100;
            }
            v2 = v93;
            if ( v42 != v104 )
              _libc_free((unsigned __int64)v42);
            v53 = v99;
            v96 = v95;
            v54 = *(_QWORD *)(a1 + 1008);
            v55 = &v99[(unsigned int)v100];
            v97 = sub_18AC630(v91, &v96)[1];
            v94 = sub_18AAE60(a1, &v97)[1];
            if ( v53 != v55 )
            {
              v90 = v2;
              while ( 2 )
              {
                v98 = *v53;
                v73 = sub_15CCCD0(v54, v98, v96);
                v74 = *(_QWORD *)(a1 + 1016);
                v75 = *(_DWORD *)(v74 + 24);
                if ( v75 )
                {
                  v56 = *(_QWORD *)(v74 + 8);
                  v57 = v75 - 1;
                  v58 = v57 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
                  v59 = (__int64 *)(v56 + 16LL * v58);
                  v60 = *v59;
                  if ( v96 == *v59 )
                  {
LABEL_68:
                    v61 = v59[1];
                  }
                  else
                  {
                    v78 = 1;
                    while ( v60 != -8 )
                    {
                      v86 = v78 + 1;
                      v58 = v57 & (v78 + v58);
                      v59 = (__int64 *)(v56 + 16LL * v58);
                      v60 = *v59;
                      if ( v96 == *v59 )
                        goto LABEL_68;
                      v78 = v86;
                    }
                    v61 = 0;
                  }
                  v62 = v98;
                  v63 = v57 & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
                  v64 = (__int64 *)(v56 + 16LL * v63);
                  v65 = *v64;
                  if ( *v64 == v98 )
                  {
LABEL_70:
                    v66 = v64[1] == v61;
                  }
                  else
                  {
                    v77 = 1;
                    while ( v65 != -8 )
                    {
                      v85 = v77 + 1;
                      v63 = v57 & (v77 + v63);
                      v64 = (__int64 *)(v56 + 16LL * v63);
                      v65 = *v64;
                      if ( *v64 == v98 )
                        goto LABEL_70;
                      v77 = v85;
                    }
                    v66 = v61 == 0;
                  }
                }
                else
                {
                  v62 = v98;
                  v66 = 1;
                }
                if ( v96 == v62 || !v73 || !v66 )
                  goto LABEL_81;
                v67 = *(_DWORD *)(a1 + 960);
                if ( v67 )
                {
                  v68 = *(_QWORD *)(a1 + 944);
                  v69 = (v67 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
                  v70 = (__int64 *)(v68 + 16LL * v69);
                  v71 = *v70;
                  if ( *v70 == v62 )
                    goto LABEL_76;
                  v81 = 1;
                  v82 = 0;
                  while ( v71 != -8 )
                  {
                    if ( !v82 && v71 == -16 )
                      v82 = v70;
                    v69 = (v67 - 1) & (v81 + v69);
                    v70 = (__int64 *)(v68 + 16LL * v69);
                    v71 = *v70;
                    if ( *v70 == v62 )
                      goto LABEL_76;
                    ++v81;
                  }
                  v83 = *(_DWORD *)(a1 + 952);
                  if ( v82 )
                    v70 = v82;
                  ++*(_QWORD *)(a1 + 936);
                  v84 = v83 + 1;
                  if ( 4 * (v83 + 1) < 3 * v67 )
                  {
                    if ( v67 - *(_DWORD *)(a1 + 956) - v84 > v67 >> 3 )
                    {
LABEL_104:
                      *(_DWORD *)(a1 + 952) = v84;
                      if ( *v70 != -8 )
                        --*(_DWORD *)(a1 + 956);
                      *v70 = v62;
                      v70[1] = 0;
                      v62 = v98;
LABEL_76:
                      v70[1] = v97;
                      if ( sub_1377F70(a1 + 64, v62) )
                        sub_1412190(a1 + 64, v97);
                      v72 = sub_18AAE60(a1, &v98)[1];
                      if ( v94 >= v72 )
                        v72 = v94;
                      v94 = v72;
LABEL_81:
                      if ( v55 == ++v53 )
                      {
                        v2 = v90;
                        goto LABEL_85;
                      }
                      continue;
                    }
                    sub_18AC470(v91, v67);
LABEL_109:
                    sub_18A8880(v91, &v98, &v102);
                    v70 = v102;
                    v62 = v98;
                    v84 = *(_DWORD *)(a1 + 952) + 1;
                    goto LABEL_104;
                  }
                }
                else
                {
                  ++*(_QWORD *)(a1 + 936);
                }
                break;
              }
              sub_18AC470(v91, 2 * v67);
              goto LABEL_109;
            }
            goto LABEL_85;
          }
        }
      }
      else
      {
        v79 = 1;
        while ( v39 != -8 )
        {
          v88 = v79 + 1;
          v37 = v36 & (v79 + v37);
          v38 = (__int64 *)(v35 + 16LL * v37);
          v39 = *v38;
          if ( *v38 == v10 )
            goto LABEL_50;
          v79 = v88;
        }
      }
    }
    v96 = v10;
    v97 = sub_18AC630(v91, &v96)[1];
    v94 = sub_18AAE60(a1, &v97)[1];
LABEL_85:
    v76 = *(_QWORD *)(*(_QWORD *)(v97 + 56) + 80LL);
    if ( v76 && v97 == v76 - 24 )
    {
      v80 = *(_QWORD *)(*(_QWORD *)(a1 + 1200) + 24LL);
      sub_18AAE60(a1, &v97)[1] = v80 + 1;
    }
    else
    {
      sub_18AAE60(a1, &v97)[1] = v94;
    }
LABEL_4:
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v92 != v2 );
  v16 = v2;
  v17 = *(_QWORD *)(a2 + 80);
  if ( v92 != v17 )
  {
    while ( 1 )
    {
      v24 = v17 - 24;
      v25 = *(_DWORD *)(a1 + 960);
      if ( !v17 )
        v24 = 0;
      v98 = v24;
      if ( !v25 )
        break;
      v18 = *(_QWORD *)(a1 + 944);
      v19 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v24 != *v20 )
      {
        v28 = 1;
        v26 = 0;
        while ( v21 != -8 )
        {
          if ( v26 || v21 != -16 )
            v20 = v26;
          v19 = (v25 - 1) & (v28 + v19);
          v87 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v87;
          if ( v24 == *v87 )
          {
            v22 = v87[1];
            goto LABEL_18;
          }
          ++v28;
          v26 = v20;
          v20 = (__int64 *)(v18 + 16LL * v19);
        }
        v29 = *(_DWORD *)(a1 + 952);
        if ( !v26 )
          v26 = v20;
        ++*(_QWORD *)(a1 + 936);
        v27 = v29 + 1;
        if ( 4 * (v29 + 1) < 3 * v25 )
        {
          if ( v25 - *(_DWORD *)(a1 + 956) - v27 <= v25 >> 3 )
            goto LABEL_26;
LABEL_33:
          *(_DWORD *)(a1 + 952) = v27;
          if ( *v26 != -8 )
            --*(_DWORD *)(a1 + 956);
          *v26 = v24;
          v22 = 0;
          v26[1] = 0;
          v21 = v98;
          goto LABEL_18;
        }
LABEL_25:
        v25 *= 2;
LABEL_26:
        sub_18AC470(v91, v25);
        sub_18A8880(v91, &v98, &v102);
        v26 = v102;
        v24 = v98;
        v27 = *(_DWORD *)(a1 + 952) + 1;
        goto LABEL_33;
      }
      v22 = v20[1];
LABEL_18:
      v102 = (__int64 *)v22;
      if ( v21 != v22 )
      {
        v23 = sub_18AAE60(a1, (__int64 *)&v102);
        sub_18AAE60(a1, &v98)[1] = v23[1];
      }
      v17 = *(_QWORD *)(v17 + 8);
      if ( v16 == v17 )
        goto LABEL_11;
    }
    ++*(_QWORD *)(a1 + 936);
    goto LABEL_25;
  }
LABEL_11:
  if ( v99 != (__int64 *)v101 )
    _libc_free((unsigned __int64)v99);
}
