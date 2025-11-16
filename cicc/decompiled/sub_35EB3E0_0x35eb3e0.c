// Function: sub_35EB3E0
// Address: 0x35eb3e0
//
__int64 __fastcall sub_35EB3E0(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r11
  __int64 v13; // r15
  int v14; // eax
  int v15; // r12d
  int v16; // r10d
  int v17; // r14d
  unsigned int v18; // edi
  _DWORD *v19; // rdx
  _DWORD *v20; // rax
  int v21; // ecx
  __int64 v22; // r12
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // r12
  int v29; // eax
  int v30; // r14d
  int v31; // r11d
  int v32; // ecx
  unsigned int v33; // r8d
  _DWORD *v34; // rdx
  _DWORD *v35; // rax
  int v36; // edi
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rdx
  unsigned int v41; // edi
  __int64 v42; // r12
  __int64 v43; // r14
  int v44; // r13d
  __int64 v45; // rcx
  int v46; // r11d
  __int64 v47; // rsi
  unsigned int v48; // r8d
  __int64 v49; // rdx
  int v50; // r9d
  __int64 *v51; // rbx
  __int64 *v52; // r11
  __int64 v53; // r9
  unsigned int v54; // r8d
  __int64 *v55; // rax
  __int64 v56; // rdi
  unsigned int v57; // esi
  int v58; // ecx
  __int64 v59; // r12
  int v60; // edi
  int v61; // edi
  __int64 v62; // r9
  unsigned int v63; // esi
  int v64; // eax
  __int64 *v65; // rdx
  __int64 v66; // r8
  __int64 v67; // rbx
  __int64 v68; // r13
  unsigned __int64 v69; // rdi
  int v71; // ecx
  int v72; // edx
  int v73; // eax
  int v74; // esi
  int v75; // esi
  __int64 v76; // r8
  __int64 *v77; // r10
  unsigned int v78; // r15d
  int v79; // r9d
  __int64 v80; // rdi
  int v81; // edx
  int v82; // ebx
  __int64 v83; // rsi
  int v84; // edi
  int v85; // r10d
  _DWORD *v86; // r9
  __int64 v87; // rsi
  int v88; // edi
  int v89; // r11d
  _DWORD *v90; // r9
  int v91; // r9d
  __int64 v92; // r14
  _DWORD *v93; // r8
  int v94; // esi
  int v95; // r11d
  __int64 v96; // rcx
  int v97; // edi
  int v98; // r15d
  __int64 *v100; // [rsp+0h] [rbp-70h]
  __int64 *v101; // [rsp+0h] [rbp-70h]
  __int64 v102; // [rsp+8h] [rbp-68h]
  __int64 v103; // [rsp+8h] [rbp-68h]
  int v104; // [rsp+8h] [rbp-68h]
  int v105; // [rsp+8h] [rbp-68h]
  int v106; // [rsp+8h] [rbp-68h]
  __int64 v107; // [rsp+10h] [rbp-60h]
  int v108; // [rsp+10h] [rbp-60h]
  __int64 v109; // [rsp+10h] [rbp-60h]
  int v110; // [rsp+10h] [rbp-60h]
  __int64 v112; // [rsp+18h] [rbp-58h]
  __int64 v113; // [rsp+20h] [rbp-50h] BYREF
  __int64 v114; // [rsp+28h] [rbp-48h]
  __int64 v115; // [rsp+30h] [rbp-40h]
  unsigned int v116; // [rsp+38h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 6436);
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v7 = sub_35E71C0(a2, a3, v6);
  v8 = *(_QWORD *)(a2 + 24);
  v102 = v7;
  v107 = v9;
  v10 = sub_2E311E0(v8);
  v11 = *(_QWORD *)(v8 + 56);
  v12 = a1;
  if ( v11 != v10 )
  {
    v13 = v10;
    while ( 1 )
    {
      v14 = sub_35E8960(a2, v11);
      v15 = v14;
      if ( !v116 )
        break;
      v16 = 1;
      v17 = 37 * v14;
      v18 = (v116 - 1) & (37 * v14);
      v19 = (_DWORD *)(v114 + 72LL * v18);
      v20 = 0;
      v21 = *v19;
      if ( v15 != *v19 )
      {
        while ( v21 != 0x7FFFFFFF )
        {
          if ( v21 == 0x80000000 && !v20 )
            v20 = v19;
          v18 = (v116 - 1) & (v16 + v18);
          v19 = (_DWORD *)(v114 + 72LL * v18);
          v21 = *v19;
          if ( v15 == *v19 )
            goto LABEL_6;
          ++v16;
        }
        if ( !v20 )
          v20 = v19;
        ++v113;
        v71 = v115 + 1;
        if ( 4 * ((int)v115 + 1) < 3 * v116 )
        {
          if ( v116 - HIDWORD(v115) - v71 <= v116 >> 3 )
          {
            sub_35EB0B0((__int64)&v113, v116);
            if ( !v116 )
            {
LABEL_167:
              LODWORD(v115) = v115 + 1;
              BUG();
            }
            v91 = 1;
            LODWORD(v92) = (v116 - 1) & v17;
            v93 = 0;
            v71 = v115 + 1;
            v20 = (_DWORD *)(v114 + 72LL * (unsigned int)v92);
            v94 = *v20;
            if ( v15 != *v20 )
            {
              while ( v94 != 0x7FFFFFFF )
              {
                if ( !v93 && v94 == 0x80000000 )
                  v93 = v20;
                v92 = (v116 - 1) & ((_DWORD)v92 + v91);
                v20 = (_DWORD *)(v114 + 72 * v92);
                v94 = *v20;
                if ( v15 == *v20 )
                  goto LABEL_65;
                ++v91;
              }
              if ( v93 )
                v20 = v93;
            }
          }
          goto LABEL_65;
        }
LABEL_101:
        sub_35EB0B0((__int64)&v113, 2 * v116);
        if ( !v116 )
          goto LABEL_167;
        LODWORD(v83) = (v116 - 1) & (37 * v15);
        v71 = v115 + 1;
        v20 = (_DWORD *)(v114 + 72LL * (unsigned int)v83);
        v84 = *v20;
        if ( v15 != *v20 )
        {
          v85 = 1;
          v86 = 0;
          while ( v84 != 0x7FFFFFFF )
          {
            if ( !v86 && v84 == 0x80000000 )
              v86 = v20;
            v83 = (v116 - 1) & ((_DWORD)v83 + v85);
            v20 = (_DWORD *)(v114 + 72 * v83);
            v84 = *v20;
            if ( v15 == *v20 )
              goto LABEL_65;
            ++v85;
          }
          if ( v86 )
            v20 = v86;
        }
LABEL_65:
        LODWORD(v115) = v71;
        if ( *v20 != 0x7FFFFFFF )
          --HIDWORD(v115);
        *v20 = v15;
        v22 = (__int64)(v20 + 2);
        *((_QWORD *)v20 + 1) = v20 + 6;
        *((_QWORD *)v20 + 2) = 0x600000000LL;
        goto LABEL_7;
      }
LABEL_6:
      v22 = (__int64)(v19 + 2);
LABEL_7:
      v25 = sub_35E86F0(a2, v11);
      v26 = *(unsigned int *)(v22 + 8);
      if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 12) )
      {
        sub_C8D5F0(v22, (const void *)(v22 + 16), v26 + 1, 8u, v23, v24);
        v26 = *(unsigned int *)(v22 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v22 + 8 * v26) = v25;
      ++*(_DWORD *)(v22 + 8);
      if ( !v11 )
        BUG();
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v13 == v11 )
          goto LABEL_12;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v13 == v11 )
        {
LABEL_12:
          v12 = a1;
          goto LABEL_13;
        }
      }
    }
    ++v113;
    goto LABEL_101;
  }
LABEL_13:
  v27 = v102;
  if ( v102 == v107 )
    goto LABEL_25;
  v103 = v12;
  v28 = v107;
  do
  {
    while ( 1 )
    {
      v29 = sub_35E8960(a2, v27);
      v30 = v29;
      if ( v116 )
      {
        v31 = 1;
        v32 = 37 * v29;
        v33 = (v116 - 1) & (37 * v29);
        v34 = (_DWORD *)(v114 + 72LL * v33);
        v35 = 0;
        v36 = *v34;
        if ( v30 == *v34 )
        {
LABEL_18:
          v37 = (__int64)(v34 + 2);
          goto LABEL_19;
        }
        while ( v36 != 0x7FFFFFFF )
        {
          if ( v36 == 0x80000000 && !v35 )
            v35 = v34;
          v33 = (v116 - 1) & (v31 + v33);
          v34 = (_DWORD *)(v114 + 72LL * v33);
          v36 = *v34;
          if ( v30 == *v34 )
            goto LABEL_18;
          ++v31;
        }
        if ( !v35 )
          v35 = v34;
        ++v113;
        v72 = v115 + 1;
        if ( 4 * ((int)v115 + 1) < 3 * v116 )
        {
          if ( v116 - HIDWORD(v115) - v72 > v116 >> 3 )
            goto LABEL_81;
          v110 = v32;
          sub_35EB0B0((__int64)&v113, v116);
          if ( !v116 )
          {
LABEL_166:
            LODWORD(v115) = v115 + 1;
            BUG();
          }
          v95 = 1;
          v90 = 0;
          LODWORD(v96) = (v116 - 1) & v110;
          v72 = v115 + 1;
          v35 = (_DWORD *)(v114 + 72LL * (unsigned int)v96);
          v97 = *v35;
          if ( v30 == *v35 )
            goto LABEL_81;
          while ( v97 != 0x7FFFFFFF )
          {
            if ( v97 == 0x80000000 && !v90 )
              v90 = v35;
            v96 = (v116 - 1) & ((_DWORD)v96 + v95);
            v35 = (_DWORD *)(v114 + 72 * v96);
            v97 = *v35;
            if ( v30 == *v35 )
              goto LABEL_81;
            ++v95;
          }
          goto LABEL_113;
        }
      }
      else
      {
        ++v113;
      }
      sub_35EB0B0((__int64)&v113, 2 * v116);
      if ( !v116 )
        goto LABEL_166;
      LODWORD(v87) = (v116 - 1) & (37 * v30);
      v72 = v115 + 1;
      v35 = (_DWORD *)(v114 + 72LL * (unsigned int)v87);
      v88 = *v35;
      if ( v30 == *v35 )
        goto LABEL_81;
      v89 = 1;
      v90 = 0;
      while ( v88 != 0x7FFFFFFF )
      {
        if ( !v90 && v88 == 0x80000000 )
          v90 = v35;
        v87 = (v116 - 1) & ((_DWORD)v87 + v89);
        v35 = (_DWORD *)(v114 + 72 * v87);
        v88 = *v35;
        if ( v30 == *v35 )
          goto LABEL_81;
        ++v89;
      }
LABEL_113:
      if ( v90 )
        v35 = v90;
LABEL_81:
      LODWORD(v115) = v72;
      if ( *v35 != 0x7FFFFFFF )
        --HIDWORD(v115);
      *v35 = v30;
      v37 = (__int64)(v35 + 2);
      *((_QWORD *)v35 + 1) = v35 + 6;
      *((_QWORD *)v35 + 2) = 0x600000000LL;
LABEL_19:
      v38 = sub_35E86F0(a2, v27);
      v40 = *(unsigned int *)(v37 + 8);
      if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 12) )
      {
        v109 = v38;
        sub_C8D5F0(v37, (const void *)(v37 + 16), v40 + 1, 8u, v40 + 1, v39);
        v40 = *(unsigned int *)(v37 + 8);
        v38 = v109;
      }
      *(_QWORD *)(*(_QWORD *)v37 + 8 * v40) = v38;
      ++*(_DWORD *)(v37 + 8);
      if ( !v27 )
        BUG();
      if ( (*(_BYTE *)v27 & 4) == 0 )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( v28 == v27 )
        goto LABEL_24;
    }
    while ( (*(_BYTE *)(v27 + 44) & 8) != 0 )
      v27 = *(_QWORD *)(v27 + 8);
    v27 = *(_QWORD *)(v27 + 8);
  }
  while ( v28 != v27 );
LABEL_24:
  v12 = v103;
LABEL_25:
  *(_QWORD *)v12 = 0;
  v41 = v116;
  v42 = v114;
  *(_QWORD *)(v12 + 8) = 0;
  *(_QWORD *)(v12 + 16) = 0;
  *(_DWORD *)(v12 + 24) = 0;
  if ( a4 <= 0 )
  {
    v47 = 72LL * v41;
  }
  else
  {
    v43 = v12;
    v44 = 0;
    v45 = v41;
    v46 = 0;
    do
    {
      v47 = 72 * v45;
      if ( !v41 )
        goto LABEL_42;
      v48 = (v41 - 1) & (37 * v46);
      v49 = v42 + 72LL * v48;
      v50 = *(_DWORD *)v49;
      if ( v46 == *(_DWORD *)v49 )
      {
LABEL_29:
        if ( v49 != v42 + 72 * v45 )
        {
          v51 = *(__int64 **)(v49 + 8);
          if ( &v51[*(unsigned int *)(v49 + 16)] != v51 )
          {
            v108 = v46;
            v52 = &v51[*(unsigned int *)(v49 + 16)];
            while ( 1 )
            {
              v57 = *(_DWORD *)(v43 + 24);
              v58 = v44;
              v59 = *v51;
              ++v44;
              if ( !v57 )
                break;
              v53 = *(_QWORD *)(v43 + 8);
              v54 = (v57 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
              v55 = (__int64 *)(v53 + 16LL * v54);
              v56 = *v55;
              if ( v59 == *v55 )
              {
LABEL_33:
                ++v51;
                *((_DWORD *)v55 + 2) = v58;
                if ( v52 == v51 )
                  goto LABEL_41;
              }
              else
              {
                v105 = 1;
                v65 = 0;
                while ( v56 != -4096 )
                {
                  if ( v56 == -8192 && !v65 )
                    v65 = v55;
                  v54 = (v57 - 1) & (v105 + v54);
                  v55 = (__int64 *)(v53 + 16LL * v54);
                  v56 = *v55;
                  if ( v59 == *v55 )
                    goto LABEL_33;
                  ++v105;
                }
                if ( !v65 )
                  v65 = v55;
                v73 = *(_DWORD *)(v43 + 16);
                ++*(_QWORD *)v43;
                v64 = v73 + 1;
                if ( 4 * v64 < 3 * v57 )
                {
                  if ( v57 - *(_DWORD *)(v43 + 20) - v64 > v57 >> 3 )
                    goto LABEL_38;
                  v101 = v52;
                  v106 = v58;
                  sub_354C5D0(v43, v57);
                  v74 = *(_DWORD *)(v43 + 24);
                  if ( !v74 )
                  {
LABEL_165:
                    ++*(_DWORD *)(v43 + 16);
                    BUG();
                  }
                  v75 = v74 - 1;
                  v76 = *(_QWORD *)(v43 + 8);
                  v77 = 0;
                  v78 = v75 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
                  v58 = v106;
                  v52 = v101;
                  v79 = 1;
                  v64 = *(_DWORD *)(v43 + 16) + 1;
                  v65 = (__int64 *)(v76 + 16LL * v78);
                  v80 = *v65;
                  if ( v59 == *v65 )
                    goto LABEL_38;
                  while ( v80 != -4096 )
                  {
                    if ( v80 == -8192 && !v77 )
                      v77 = v65;
                    v78 = v75 & (v79 + v78);
                    v65 = (__int64 *)(v76 + 16LL * v78);
                    v80 = *v65;
                    if ( v59 == *v65 )
                      goto LABEL_38;
                    ++v79;
                  }
                  goto LABEL_93;
                }
LABEL_36:
                v100 = v52;
                v104 = v58;
                sub_354C5D0(v43, 2 * v57);
                v60 = *(_DWORD *)(v43 + 24);
                if ( !v60 )
                  goto LABEL_165;
                v61 = v60 - 1;
                v62 = *(_QWORD *)(v43 + 8);
                v58 = v104;
                v52 = v100;
                v63 = v61 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
                v64 = *(_DWORD *)(v43 + 16) + 1;
                v65 = (__int64 *)(v62 + 16LL * v63);
                v66 = *v65;
                if ( v59 == *v65 )
                  goto LABEL_38;
                v98 = 1;
                v77 = 0;
                while ( v66 != -4096 )
                {
                  if ( !v77 && v66 == -8192 )
                    v77 = v65;
                  v63 = v61 & (v98 + v63);
                  v65 = (__int64 *)(v62 + 16LL * v63);
                  v66 = *v65;
                  if ( v59 == *v65 )
                    goto LABEL_38;
                  ++v98;
                }
LABEL_93:
                if ( v77 )
                  v65 = v77;
LABEL_38:
                *(_DWORD *)(v43 + 16) = v64;
                if ( *v65 != -4096 )
                  --*(_DWORD *)(v43 + 20);
                ++v51;
                *v65 = v59;
                *((_DWORD *)v65 + 2) = 0;
                *((_DWORD *)v65 + 2) = v58;
                if ( v52 == v51 )
                {
LABEL_41:
                  v45 = v116;
                  v46 = v108;
                  v42 = v114;
                  v41 = v116;
                  v47 = 72LL * v116;
                  goto LABEL_42;
                }
              }
            }
            ++*(_QWORD *)v43;
            goto LABEL_36;
          }
        }
      }
      else
      {
        v81 = 1;
        while ( v50 != 0x7FFFFFFF )
        {
          v82 = v81 + 1;
          v48 = (v41 - 1) & (v81 + v48);
          v49 = v42 + 72LL * v48;
          v50 = *(_DWORD *)v49;
          if ( v46 == *(_DWORD *)v49 )
            goto LABEL_29;
          v81 = v82;
        }
      }
LABEL_42:
      ++v46;
    }
    while ( v46 != a4 );
    v12 = v43;
  }
  if ( v41 )
  {
    v67 = v42 + v47;
    v68 = v12;
    do
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)v42 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        {
          v69 = *(_QWORD *)(v42 + 8);
          if ( v69 != v42 + 24 )
            break;
        }
        v42 += 72;
        if ( v67 == v42 )
          goto LABEL_50;
      }
      _libc_free(v69);
      v42 += 72;
    }
    while ( v67 != v42 );
LABEL_50:
    v42 = v114;
    v12 = v68;
    v47 = 72LL * v116;
  }
  v112 = v12;
  sub_C7D6A0(v42, v47, 8);
  return v112;
}
