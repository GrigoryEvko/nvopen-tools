// Function: sub_23C0920
// Address: 0x23c0920
//
__int64 __fastcall sub_23C0920(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  int v6; // edx
  unsigned __int64 v7; // r12
  __int64 v8; // r12
  int v9; // r15d
  _QWORD *v10; // rax
  _QWORD *v11; // r14
  _QWORD *v12; // rcx
  __int64 v13; // rax
  int v14; // r13d
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 result; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // rdi
  int v24; // r9d
  unsigned int v25; // ecx
  __int64 *v26; // r14
  __int64 *v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // r10
  __int64 v31; // rcx
  int v32; // r11d
  __int64 *v33; // rdx
  unsigned int v34; // r8d
  __int64 *v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 v39; // rbx
  int v40; // ecx
  int v41; // ecx
  __int64 v42; // r8
  unsigned int v43; // esi
  int v44; // edx
  __int64 v45; // rdi
  int v46; // esi
  int v47; // esi
  __int64 v48; // r9
  unsigned int v49; // eax
  int v50; // edi
  __int64 v51; // r8
  _DWORD *v52; // rdx
  unsigned int v53; // esi
  __int64 v54; // rdi
  unsigned int v55; // ecx
  __int64 *v56; // rdx
  __int64 v57; // rax
  int v58; // r11d
  __int64 *v59; // r14
  int v60; // eax
  int v61; // eax
  int v62; // eax
  int v63; // eax
  int v64; // eax
  __int64 v65; // r8
  __int64 *v66; // r9
  unsigned int v67; // r15d
  int v68; // r11d
  __int64 v69; // rsi
  int v70; // edi
  int v71; // ecx
  int v72; // ecx
  __int64 v73; // r8
  int v74; // r11d
  __int64 *v75; // r10
  unsigned int v76; // esi
  __int64 v77; // rdi
  unsigned int v78; // esi
  __int64 v79; // rbx
  __int64 v80; // rdi
  __int64 v81; // r8
  unsigned int v82; // eax
  __int64 *v83; // rdx
  __int64 v84; // rcx
  int v85; // r10d
  __int64 *v86; // rbx
  int v87; // eax
  int v88; // edx
  int v89; // edx
  int v90; // edx
  __int64 v91; // rdi
  unsigned int v92; // ecx
  __int64 v93; // rsi
  int v94; // r10d
  __int64 *v95; // r9
  int v96; // edx
  int v97; // edx
  __int64 v98; // rsi
  int v99; // r9d
  __int64 *v100; // r8
  unsigned int v101; // r15d
  __int64 v102; // rcx
  int v103; // r14d
  __int64 *v104; // r11
  int v105; // r11d
  __int64 v106; // [rsp+0h] [rbp-80h]
  __int64 v107; // [rsp+8h] [rbp-78h]
  __int64 v108; // [rsp+10h] [rbp-70h]
  __int64 v109; // [rsp+18h] [rbp-68h]
  unsigned int v110; // [rsp+24h] [rbp-5Ch]
  __int64 v111; // [rsp+28h] [rbp-58h]
  int v112; // [rsp+30h] [rbp-50h]
  __int64 v114; // [rsp+38h] [rbp-48h]
  __int64 v115; // [rsp+40h] [rbp-40h] BYREF
  __int64 *v116; // [rsp+48h] [rbp-38h] BYREF

  v3 = a1;
  v4 = a2;
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  v107 = a1 + 40;
  v108 = a2 + 72;
  if ( !a3 )
    goto LABEL_19;
  v5 = *(_QWORD *)(a2 + 80);
  v6 = 0;
  if ( v5 == a2 + 72 )
    goto LABEL_123;
  do
  {
    v5 = *(_QWORD *)(v5 + 8);
    ++v6;
  }
  while ( v5 != v108 );
  if ( !v6 )
  {
LABEL_123:
    v9 = 0;
    v11 = 0;
LABEL_124:
    *(_QWORD *)(a1 + 8) = v11;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = v9;
    *(_BYTE *)(a1 + 32) = 1;
    sub_C7D6A0(0, 0, 8);
    goto LABEL_19;
  }
  v7 = (((((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
         | (4 * v6 / 3u + 1)
         | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
       | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
       | (4 * v6 / 3u + 1)
       | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 8)
     | (((((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
       | (4 * v6 / 3u + 1)
       | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 4)
     | (((4 * v6 / 3u + 1) | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1)) >> 2)
     | (4 * v6 / 3u + 1)
     | ((unsigned __int64)(4 * v6 / 3u + 1) >> 1);
  v8 = ((v7 >> 16) | v7) + 1;
  v9 = v8;
  v10 = (_QWORD *)sub_C7D670(40 * v8, 8);
  v11 = v10;
  v12 = &v10[5 * v8];
  do
  {
    if ( v10 )
      *v10 = 0x7FFFFFFFFFFFFFFFLL;
    v10 += 5;
  }
  while ( v12 != v10 );
  if ( !*(_BYTE *)(a1 + 32) )
    goto LABEL_124;
  v13 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v13 )
  {
    v14 = v8;
    v15 = *(__int64 **)(a1 + 8);
    v16 = &v15[5 * v13];
    do
    {
      while ( 1 )
      {
        if ( *v15 <= 0x7FFFFFFFFFFFFFFDLL )
        {
          v15[1] = (__int64)&unk_49DB368;
          v17 = v15[4];
          if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
            break;
        }
        v15 += 5;
        if ( v16 == v15 )
          goto LABEL_17;
      }
      v18 = v15 + 2;
      v15 += 5;
      sub_BD60C0(v18);
    }
    while ( v16 != v15 );
LABEL_17:
    LODWORD(v8) = v14;
    v3 = a1;
    v4 = a2;
    v13 = *(unsigned int *)(a1 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(v3 + 8), 40 * v13, 8);
  ++*(_QWORD *)v3;
  *(_QWORD *)(v3 + 8) = v11;
  *(_QWORD *)(v3 + 16) = 0;
  *(_DWORD *)(v3 + 24) = v8;
  sub_C7D6A0(0, 0, 8);
LABEL_19:
  result = *(_QWORD *)(v4 + 80);
  v109 = result;
  if ( result != v108 )
  {
    while ( 1 )
    {
      v20 = 0;
      if ( v109 )
        v20 = v109 - 24;
      v114 = v20;
      if ( *(_BYTE *)(v3 + 32) )
        break;
LABEL_23:
      v21 = *(_QWORD *)(v114 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 != v114 + 48 )
      {
        if ( !v21 )
          BUG();
        v111 = v21 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v21 - 24) - 30 <= 0xA )
        {
          v112 = sub_B46E30(v21 - 24);
          if ( v112 )
          {
            v22 = 0;
            v110 = ((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4);
            while ( 1 )
            {
              v37 = sub_B46EC0(v111, v22);
              v38 = *(_DWORD *)(v3 + 64);
              v39 = v37;
              if ( !v38 )
                break;
              v23 = *(_QWORD *)(v3 + 48);
              v24 = 1;
              v25 = (v38 - 1) & v110;
              v26 = (__int64 *)(v23 + 40LL * v25);
              v27 = 0;
              v28 = *v26;
              if ( v114 != *v26 )
              {
                while ( v28 != -4096 )
                {
                  if ( v28 == -8192 && !v27 )
                    v27 = v26;
                  v25 = (v38 - 1) & (v24 + v25);
                  v26 = (__int64 *)(v23 + 40LL * v25);
                  v28 = *v26;
                  if ( v114 == *v26 )
                    goto LABEL_29;
                  ++v24;
                }
                v70 = *(_DWORD *)(v3 + 56);
                if ( !v27 )
                  v27 = v26;
                ++*(_QWORD *)(v3 + 40);
                v44 = v70 + 1;
                if ( 4 * (v70 + 1) < 3 * v38 )
                {
                  if ( v38 - *(_DWORD *)(v3 + 60) - v44 <= v38 >> 3 )
                  {
                    sub_23C06A0(v107, v38);
                    v71 = *(_DWORD *)(v3 + 64);
                    if ( !v71 )
                    {
LABEL_177:
                      ++*(_DWORD *)(v3 + 56);
                      BUG();
                    }
                    v72 = v71 - 1;
                    v73 = *(_QWORD *)(v3 + 48);
                    v74 = 1;
                    v75 = 0;
                    v76 = v72 & v110;
                    v44 = *(_DWORD *)(v3 + 56) + 1;
                    v27 = (__int64 *)(v73 + 40LL * (v72 & v110));
                    v77 = *v27;
                    if ( v114 != *v27 )
                    {
                      while ( v77 != -4096 )
                      {
                        if ( v77 == -8192 && !v75 )
                          v75 = v27;
                        v76 = v72 & (v74 + v76);
                        v27 = (__int64 *)(v73 + 40LL * v76);
                        v77 = *v27;
                        if ( v114 == *v27 )
                          goto LABEL_37;
                        ++v74;
                      }
LABEL_91:
                      if ( v75 )
                        v27 = v75;
                    }
                  }
LABEL_37:
                  *(_DWORD *)(v3 + 56) = v44;
                  if ( *v27 != -4096 )
                    --*(_DWORD *)(v3 + 60);
                  v27[1] = 0;
                  v31 = (__int64)(v27 + 1);
                  v27[2] = 0;
                  *v27 = v114;
                  v27[3] = 0;
                  *((_DWORD *)v27 + 8) = 0;
LABEL_40:
                  ++*(_QWORD *)v31;
                  v29 = 0;
LABEL_41:
                  v106 = v31;
                  sub_A4A350(v31, 2 * v29);
                  v31 = v106;
                  v46 = *(_DWORD *)(v106 + 24);
                  if ( !v46 )
                    goto LABEL_176;
                  v47 = v46 - 1;
                  v48 = *(_QWORD *)(v106 + 8);
                  v49 = v47 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                  v50 = *(_DWORD *)(v106 + 16) + 1;
                  v33 = (__int64 *)(v48 + 16LL * v49);
                  v51 = *v33;
                  if ( v39 != *v33 )
                  {
                    v103 = 1;
                    v104 = 0;
                    while ( v51 != -4096 )
                    {
                      if ( v51 == -8192 && !v104 )
                        v104 = v33;
                      v49 = v47 & (v103 + v49);
                      v33 = (__int64 *)(v48 + 16LL * v49);
                      v51 = *v33;
                      if ( v39 == *v33 )
                        goto LABEL_43;
                      ++v103;
                    }
                    if ( v104 )
                      v33 = v104;
                  }
                  goto LABEL_43;
                }
LABEL_35:
                sub_23C06A0(v107, 2 * v38);
                v40 = *(_DWORD *)(v3 + 64);
                if ( !v40 )
                  goto LABEL_177;
                v41 = v40 - 1;
                v42 = *(_QWORD *)(v3 + 48);
                v43 = v41 & v110;
                v44 = *(_DWORD *)(v3 + 56) + 1;
                v27 = (__int64 *)(v42 + 40LL * (v41 & v110));
                v45 = *v27;
                if ( v114 != *v27 )
                {
                  v105 = 1;
                  v75 = 0;
                  while ( v45 != -4096 )
                  {
                    if ( v45 == -8192 && !v75 )
                      v75 = v27;
                    v43 = v41 & (v105 + v43);
                    v27 = (__int64 *)(v42 + 40LL * v43);
                    v45 = *v27;
                    if ( v114 == *v27 )
                      goto LABEL_37;
                    ++v105;
                  }
                  goto LABEL_91;
                }
                goto LABEL_37;
              }
LABEL_29:
              v29 = *((_DWORD *)v26 + 8);
              v30 = v26[2];
              v31 = (__int64)(v26 + 1);
              if ( !v29 )
                goto LABEL_40;
              v32 = 1;
              v33 = 0;
              v34 = (v29 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
              v35 = (__int64 *)(v30 + 16LL * v34);
              v36 = *v35;
              if ( v39 == *v35 )
              {
LABEL_31:
                ++*((_DWORD *)v35 + 2);
                if ( !*(_BYTE *)(v3 + 32) )
                  goto LABEL_32;
                goto LABEL_46;
              }
              while ( v36 != -4096 )
              {
                if ( v36 == -8192 && !v33 )
                  v33 = v35;
                v34 = (v29 - 1) & (v32 + v34);
                v35 = (__int64 *)(v30 + 16LL * v34);
                v36 = *v35;
                if ( v39 == *v35 )
                  goto LABEL_31;
                ++v32;
              }
              if ( !v33 )
                v33 = v35;
              v62 = *((_DWORD *)v26 + 6);
              ++v26[1];
              v50 = v62 + 1;
              if ( 4 * (v62 + 1) >= 3 * v29 )
                goto LABEL_41;
              if ( v29 - *((_DWORD *)v26 + 7) - v50 <= v29 >> 3 )
              {
                sub_A4A350((__int64)(v26 + 1), v29);
                v63 = *((_DWORD *)v26 + 8);
                v31 = (__int64)(v26 + 1);
                if ( !v63 )
                {
LABEL_176:
                  ++*(_DWORD *)(v31 + 16);
                  BUG();
                }
                v64 = v63 - 1;
                v65 = v26[2];
                v66 = 0;
                v67 = v64 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v68 = 1;
                v50 = *((_DWORD *)v26 + 6) + 1;
                v33 = (__int64 *)(v65 + 16LL * v67);
                v69 = *v33;
                if ( v39 != *v33 )
                {
                  while ( v69 != -4096 )
                  {
                    if ( v66 || v69 != -8192 )
                      v33 = v66;
                    v67 = v64 & (v68 + v67);
                    v69 = *(_QWORD *)(v65 + 16LL * v67);
                    if ( v39 == v69 )
                    {
                      v33 = (__int64 *)(v65 + 16LL * v67);
                      goto LABEL_43;
                    }
                    ++v68;
                    v66 = v33;
                    v33 = (__int64 *)(v65 + 16LL * v67);
                  }
                  if ( v66 )
                    v33 = v66;
                }
              }
LABEL_43:
              *(_DWORD *)(v31 + 16) = v50;
              if ( *v33 != -4096 )
                --*(_DWORD *)(v31 + 20);
              *v33 = v39;
              v52 = v33 + 1;
              *v52 = 0;
              *v52 = 1;
              if ( !*(_BYTE *)(v3 + 32) )
                goto LABEL_32;
LABEL_46:
              v53 = *(_DWORD *)(v3 + 24);
              if ( !v53 )
              {
                ++*(_QWORD *)v3;
                goto LABEL_110;
              }
              v54 = *(_QWORD *)(v3 + 8);
              v55 = (v53 - 1) & (37 * v39);
              v56 = (__int64 *)(v54 + 40LL * v55);
              v57 = *v56;
              if ( v39 == *v56 )
              {
LABEL_32:
                if ( v112 == ++v22 )
                  goto LABEL_60;
              }
              else
              {
                v58 = 1;
                v59 = 0;
                while ( v57 != 0x7FFFFFFFFFFFFFFFLL )
                {
                  if ( v57 != 0x7FFFFFFFFFFFFFFELL || v59 )
                    v56 = v59;
                  v55 = (v53 - 1) & (v58 + v55);
                  v57 = *(_QWORD *)(v54 + 40LL * v55);
                  if ( v39 == v57 )
                    goto LABEL_32;
                  v59 = v56;
                  ++v58;
                  v56 = (__int64 *)(v54 + 40LL * v55);
                }
                v60 = *(_DWORD *)(v3 + 16);
                if ( !v59 )
                  v59 = v56;
                ++*(_QWORD *)v3;
                v61 = v60 + 1;
                if ( 4 * v61 < 3 * v53 )
                {
                  if ( v53 - *(_DWORD *)(v3 + 20) - v61 <= v53 >> 3 )
                  {
                    sub_23C03C0(v3, v53);
                    v96 = *(_DWORD *)(v3 + 24);
                    if ( !v96 )
                    {
LABEL_178:
                      ++*(_DWORD *)(v3 + 16);
                      BUG();
                    }
                    v97 = v96 - 1;
                    v98 = *(_QWORD *)(v3 + 8);
                    v99 = 1;
                    v100 = 0;
                    v101 = v97 & (37 * v39);
                    v59 = (__int64 *)(v98 + 40LL * v101);
                    v102 = *v59;
                    v61 = *(_DWORD *)(v3 + 16) + 1;
                    if ( v39 != *v59 )
                    {
                      while ( v102 != 0x7FFFFFFFFFFFFFFFLL )
                      {
                        if ( !v100 && v102 == 0x7FFFFFFFFFFFFFFELL )
                          v100 = v59;
                        v101 = v97 & (v99 + v101);
                        v59 = (__int64 *)(v98 + 40LL * v101);
                        v102 = *v59;
                        if ( v39 == *v59 )
                          goto LABEL_54;
                        ++v99;
                      }
                      if ( v100 )
                        v59 = v100;
                    }
                  }
                  goto LABEL_54;
                }
LABEL_110:
                sub_23C03C0(v3, 2 * v53);
                v89 = *(_DWORD *)(v3 + 24);
                if ( !v89 )
                  goto LABEL_178;
                v90 = v89 - 1;
                v91 = *(_QWORD *)(v3 + 8);
                v92 = v90 & (37 * v39);
                v59 = (__int64 *)(v91 + 40LL * v92);
                v93 = *v59;
                v61 = *(_DWORD *)(v3 + 16) + 1;
                if ( v39 != *v59 )
                {
                  v94 = 1;
                  v95 = 0;
                  while ( v93 != 0x7FFFFFFFFFFFFFFFLL )
                  {
                    if ( !v95 && v93 == 0x7FFFFFFFFFFFFFFELL )
                      v95 = v59;
                    v92 = v90 & (v94 + v92);
                    v59 = (__int64 *)(v91 + 40LL * v92);
                    v93 = *v59;
                    if ( v39 == *v59 )
                      goto LABEL_54;
                    ++v94;
                  }
                  if ( v95 )
                    v59 = v95;
                }
LABEL_54:
                *(_DWORD *)(v3 + 16) = v61;
                if ( *v59 != 0x7FFFFFFFFFFFFFFFLL )
                  --*(_DWORD *)(v3 + 20);
                *v59 = v39;
                v59[2] = 2;
                v59[3] = 0;
                v59[4] = v39;
                if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
                  sub_BD73F0((__int64)(v59 + 2));
                ++v22;
                v59[1] = (__int64)&unk_4A15D10;
                if ( v112 == v22 )
                  goto LABEL_60;
              }
            }
            ++*(_QWORD *)(v3 + 40);
            goto LABEL_35;
          }
        }
      }
LABEL_60:
      result = *(_QWORD *)(v109 + 8);
      v109 = result;
      if ( result == v108 )
        return result;
    }
    v78 = *(_DWORD *)(v3 + 24);
    v115 = v20;
    v79 = v20;
    v80 = v20;
    if ( v78 )
    {
      v81 = *(_QWORD *)(v3 + 8);
      v82 = (v78 - 1) & (37 * v20);
      v83 = (__int64 *)(v81 + 40LL * v82);
      v84 = *v83;
      if ( v79 == *v83 )
        goto LABEL_23;
      v85 = 1;
      v86 = 0;
      while ( v84 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v86 || v84 != 0x7FFFFFFFFFFFFFFELL )
          v83 = v86;
        v82 = (v78 - 1) & (v85 + v82);
        v84 = *(_QWORD *)(v81 + 40LL * v82);
        if ( v114 == v84 )
          goto LABEL_23;
        ++v85;
        v86 = v83;
        v83 = (__int64 *)(v81 + 40LL * v82);
      }
      v87 = *(_DWORD *)(v3 + 16);
      if ( !v86 )
        v86 = v83;
      ++*(_QWORD *)v3;
      v88 = v87 + 1;
      v116 = v86;
      if ( 4 * (v87 + 1) < 3 * v78 )
      {
        if ( v78 - *(_DWORD *)(v3 + 20) - v88 > v78 >> 3 )
        {
LABEL_102:
          *(_DWORD *)(v3 + 16) = v88;
          if ( *v86 != 0x7FFFFFFFFFFFFFFFLL )
            --*(_DWORD *)(v3 + 20);
          *v86 = v80;
          v86[2] = 2;
          v86[3] = 0;
          v86[4] = v114;
          if ( v114 != -4096 && v114 != 0 && v114 != -8192 )
            sub_BD73F0((__int64)(v86 + 2));
          v86[1] = (__int64)&unk_4A15D10;
          goto LABEL_23;
        }
LABEL_139:
        sub_23C03C0(v3, v78);
        sub_23B7200(v3, &v115, &v116);
        v80 = v115;
        v86 = v116;
        v88 = *(_DWORD *)(v3 + 16) + 1;
        goto LABEL_102;
      }
    }
    else
    {
      ++*(_QWORD *)v3;
      v116 = 0;
    }
    v78 *= 2;
    goto LABEL_139;
  }
  return result;
}
