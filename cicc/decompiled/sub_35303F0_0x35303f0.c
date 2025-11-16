// Function: sub_35303F0
// Address: 0x35303f0
//
__int64 __fastcall sub_35303F0(__int64 a1)
{
  __int64 v2; // r13
  int v3; // edi
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rbx
  int v10; // r11d
  __int64 *v11; // rdx
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // r9
  _DWORD *v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 *v19; // rcx
  int v20; // eax
  __int64 *v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r9
  int v27; // r14d
  __int64 *v28; // rcx
  int v29; // ebx
  unsigned int v30; // r13d
  __int64 *v31; // r10
  unsigned int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdi
  int v35; // eax
  int v36; // edx
  __int64 v37; // rsi
  int v38; // r11d
  __int64 v39; // rdx
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // r8
  int v48; // eax
  int v49; // edx
  int v50; // r8d
  int v51; // eax
  int v52; // edi
  __int64 *v53; // rcx
  __int64 *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // r12
  __int64 *v60; // rbx
  int v61; // r14d
  __int64 *v62; // r10
  int v63; // edi
  __int64 *v64; // rdx
  __int64 v65; // rcx
  int v66; // r11d
  __int64 *v67; // rax
  int v68; // edx
  int v69; // ecx
  unsigned int v70; // esi
  __int64 v71; // r8
  int v72; // eax
  __int64 v73; // rdi
  __int64 v74; // r9
  unsigned int v75; // r13d
  int v76; // eax
  __int64 v77; // rsi
  int v78; // ecx
  unsigned int v79; // esi
  __int64 v80; // r9
  int v81; // eax
  __int64 *v82; // rdi
  unsigned int v83; // r13d
  int v84; // eax
  __int64 v85; // rsi
  int v86; // ecx
  __int64 v87; // rdi
  __int64 *v88; // rsi
  int v89; // r15d
  __int64 v90; // rcx
  int v91; // r10d
  __int64 v92; // rax
  int v93; // [rsp+4h] [rbp-ACh]
  __int64 *v94; // [rsp+8h] [rbp-A8h]
  __int64 *v95; // [rsp+8h] [rbp-A8h]
  __int64 *v96; // [rsp+18h] [rbp-98h] BYREF
  __int64 v97; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v98; // [rsp+28h] [rbp-88h]
  __int64 v99; // [rsp+30h] [rbp-80h]
  __int64 v100; // [rsp+38h] [rbp-78h]
  __int64 v101; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v102; // [rsp+48h] [rbp-68h]
  __int64 v103; // [rsp+50h] [rbp-60h]
  __int64 v104; // [rsp+58h] [rbp-58h]
  __int64 v105; // [rsp+60h] [rbp-50h] BYREF
  __int64 v106; // [rsp+68h] [rbp-48h]
  __int64 v107; // [rsp+70h] [rbp-40h]
  unsigned int v108; // [rsp+78h] [rbp-38h]

  v96 = &v101;
  v2 = *(_QWORD *)(a1 + 328);
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 1;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  sub_352FC80((__int64)&v105, 0);
  if ( !v108 )
  {
    LODWORD(v107) = v107 + 1;
    BUG();
  }
  v3 = 1;
  v4 = 0;
  v5 = (v108 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v6 = v106 + 16LL * v5;
  v7 = *(_QWORD *)v6;
  if ( v2 != *(_QWORD *)v6 )
  {
    while ( v7 != -4096 )
    {
      if ( !v4 && v7 == -8192 )
        v4 = v6;
      v5 = (v108 - 1) & (v3 + v5);
      v6 = v106 + 16LL * v5;
      v7 = *(_QWORD *)v6;
      if ( v2 == *(_QWORD *)v6 )
        goto LABEL_3;
      ++v3;
    }
    if ( v4 )
      v6 = v4;
  }
LABEL_3:
  LODWORD(v107) = v107 + 1;
  if ( *(_QWORD *)v6 != -4096 )
    --HIDWORD(v107);
  *(_QWORD *)v6 = v2;
  v8 = a1 + 320;
  *(_DWORD *)(v6 + 8) = 2;
  sub_3530160((__int64 *)&v96, v2);
  v9 = *(_QWORD *)(v8 + 8);
  if ( v9 != v8 )
  {
    while ( 1 )
    {
      while ( !*(_BYTE *)(v9 + 216) )
      {
        v9 = *(_QWORD *)(v9 + 8);
        if ( v8 == v9 )
          goto LABEL_13;
      }
      sub_3530160((__int64 *)&v96, v9);
      if ( !v108 )
        break;
      v10 = 1;
      v11 = 0;
      v12 = (v108 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v13 = (_QWORD *)(v106 + 16LL * v12);
      v14 = *v13;
      if ( *v13 != v9 )
      {
        while ( v14 != -4096 )
        {
          if ( !v11 && v14 == -8192 )
            v11 = v13;
          v12 = (v108 - 1) & (v10 + v12);
          v13 = (_QWORD *)(v106 + 16LL * v12);
          v14 = *v13;
          if ( *v13 == v9 )
            goto LABEL_11;
          ++v10;
        }
        if ( !v11 )
          v11 = v13;
        ++v105;
        v78 = v107 + 1;
        if ( 4 * ((int)v107 + 1) < 3 * v108 )
        {
          if ( v108 - HIDWORD(v107) - v78 <= v108 >> 3 )
          {
            sub_352FC80((__int64)&v105, v108);
            if ( !v108 )
            {
LABEL_208:
              LODWORD(v107) = v107 + 1;
              BUG();
            }
            v82 = 0;
            v83 = (v108 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v78 = v107 + 1;
            v84 = 1;
            v11 = (__int64 *)(v106 + 16LL * v83);
            v85 = *v11;
            if ( *v11 != v9 )
            {
              while ( v85 != -4096 )
              {
                if ( !v82 && v85 == -8192 )
                  v82 = v11;
                v91 = v84 + 1;
                v92 = (v108 - 1) & (v83 + v84);
                v83 = v92;
                v11 = (__int64 *)(v106 + 16 * v92);
                v85 = *v11;
                if ( *v11 == v9 )
                  goto LABEL_137;
                v84 = v91;
              }
              goto LABEL_153;
            }
          }
          goto LABEL_137;
        }
LABEL_141:
        sub_352FC80((__int64)&v105, 2 * v108);
        if ( !v108 )
          goto LABEL_208;
        v79 = (v108 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v11 = (__int64 *)(v106 + 16LL * v79);
        v80 = *v11;
        v78 = v107 + 1;
        if ( v9 != *v11 )
        {
          v81 = 1;
          v82 = 0;
          while ( v80 != -4096 )
          {
            if ( v80 == -8192 && !v82 )
              v82 = v11;
            v79 = (v108 - 1) & (v81 + v79);
            v11 = (__int64 *)(v106 + 16LL * v79);
            v80 = *v11;
            if ( *v11 == v9 )
              goto LABEL_137;
            ++v81;
          }
LABEL_153:
          if ( v82 )
            v11 = v82;
        }
LABEL_137:
        LODWORD(v107) = v78;
        if ( *v11 != -4096 )
          --HIDWORD(v107);
        *v11 = v9;
        v15 = v11 + 1;
        *((_DWORD *)v11 + 2) = 0;
        goto LABEL_12;
      }
LABEL_11:
      v15 = v13 + 1;
LABEL_12:
      *v15 = 1;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v8 == v9 )
        goto LABEL_13;
    }
    ++v105;
    goto LABEL_141;
  }
LABEL_13:
  if ( !(_DWORD)v103 )
    goto LABEL_44;
  do
  {
    v16 = &v102[(unsigned int)v104];
    v17 = *v102;
    if ( v102 != v16 )
    {
      v18 = v102;
      while ( 1 )
      {
        v17 = *v18;
        v19 = v18;
        if ( *v18 != -8192 && v17 != -4096 )
          break;
        if ( v16 == ++v18 )
        {
          v17 = v19[1];
          break;
        }
      }
    }
    if ( (_DWORD)v104 )
    {
      v20 = (v104 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v21 = &v102[v20];
      v22 = *v21;
      if ( v17 == *v21 )
      {
LABEL_22:
        *v21 = -8192;
        LODWORD(v103) = v103 - 1;
        ++HIDWORD(v103);
      }
      else
      {
        v49 = 1;
        while ( v22 != -4096 )
        {
          v50 = v49 + 1;
          v20 = (v104 - 1) & (v49 + v20);
          v21 = &v102[v20];
          v22 = *v21;
          if ( *v21 == v17 )
            goto LABEL_22;
          v49 = v50;
        }
      }
    }
    if ( !v108 )
      goto LABEL_50;
    v23 = (v108 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v24 = v106 + 16LL * v23;
    v25 = *(_QWORD *)v24;
    if ( v17 != *(_QWORD *)v24 )
    {
      v51 = 1;
      while ( v25 != -4096 )
      {
        v52 = v51 + 1;
        v23 = (v108 - 1) & (v51 + v23);
        v24 = v106 + 16LL * v23;
        v25 = *(_QWORD *)v24;
        if ( v17 == *(_QWORD *)v24 )
          goto LABEL_25;
        v51 = v52;
      }
LABEL_50:
      v27 = 0;
      v26 = v106 + 16LL * v108;
      goto LABEL_27;
    }
LABEL_25:
    v26 = v106 + 16LL * v108;
    if ( v26 == v24 )
      v27 = 0;
    else
      v27 = *(_DWORD *)(v24 + 8);
LABEL_27:
    v28 = *(__int64 **)(v17 + 64);
    v29 = v27;
    v30 = v108 - 1;
    v31 = &v28[*(unsigned int *)(v17 + 72)];
    if ( v28 == v31 )
      goto LABEL_13;
    do
    {
      while ( 1 )
      {
        v37 = *v28;
        if ( v108 )
          break;
        if ( v29 < 0 )
          v29 = 0;
        if ( v31 == ++v28 )
          goto LABEL_39;
      }
      v32 = v30 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v33 = v106 + 16LL * v32;
      v34 = *(_QWORD *)v33;
      if ( v37 == *(_QWORD *)v33 )
      {
LABEL_30:
        if ( v26 != v33 )
        {
          v35 = *(_DWORD *)(v33 + 8);
          v36 = v35;
          goto LABEL_32;
        }
      }
      else
      {
        v48 = 1;
        while ( v34 != -4096 )
        {
          v32 = v30 & (v48 + v32);
          v93 = v48 + 1;
          v33 = v106 + 16LL * v32;
          v34 = *(_QWORD *)v33;
          if ( v37 == *(_QWORD *)v33 )
            goto LABEL_30;
          v48 = v93;
        }
      }
      v36 = 0;
      v35 = 0;
LABEL_32:
      if ( v36 > v29 )
        v29 = v35;
      ++v28;
    }
    while ( v31 != v28 );
LABEL_39:
    if ( v29 == v27 )
      goto LABEL_13;
    sub_3530160((__int64 *)&v96, v17);
    if ( !v108 )
    {
      ++v105;
LABEL_114:
      sub_352FC80((__int64)&v105, 2 * v108);
      if ( v108 )
      {
        v70 = (v108 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v39 = v106 + 16LL * v70;
        v71 = *(_QWORD *)v39;
        v69 = v107 + 1;
        if ( v17 != *(_QWORD *)v39 )
        {
          v72 = 1;
          v73 = 0;
          while ( v71 != -4096 )
          {
            if ( v71 == -8192 && !v73 )
              v73 = v39;
            v70 = (v108 - 1) & (v72 + v70);
            v39 = v106 + 16LL * v70;
            v71 = *(_QWORD *)v39;
            if ( v17 == *(_QWORD *)v39 )
              goto LABEL_110;
            ++v72;
          }
          if ( v73 )
            v39 = v73;
        }
        goto LABEL_110;
      }
LABEL_207:
      LODWORD(v107) = v107 + 1;
      BUG();
    }
    v38 = 1;
    v39 = 0;
    v40 = (v108 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v41 = v106 + 16LL * v40;
    v42 = *(_QWORD *)v41;
    if ( v17 == *(_QWORD *)v41 )
    {
LABEL_42:
      *(_DWORD *)(v41 + 8) = v29;
      continue;
    }
    while ( v42 != -4096 )
    {
      if ( v42 == -8192 && !v39 )
        v39 = v41;
      v40 = (v108 - 1) & (v38 + v40);
      v41 = v106 + 16LL * v40;
      v42 = *(_QWORD *)v41;
      if ( v17 == *(_QWORD *)v41 )
        goto LABEL_42;
      ++v38;
    }
    if ( !v39 )
      v39 = v41;
    ++v105;
    v69 = v107 + 1;
    if ( 4 * ((int)v107 + 1) >= 3 * v108 )
      goto LABEL_114;
    if ( v108 - HIDWORD(v107) - v69 <= v108 >> 3 )
    {
      sub_352FC80((__int64)&v105, v108);
      if ( v108 )
      {
        v74 = 0;
        v75 = (v108 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v69 = v107 + 1;
        v76 = 1;
        v39 = v106 + 16LL * v75;
        v77 = *(_QWORD *)v39;
        if ( v17 != *(_QWORD *)v39 )
        {
          while ( v77 != -4096 )
          {
            if ( v77 == -8192 && !v74 )
              v74 = v39;
            v75 = (v108 - 1) & (v76 + v75);
            v39 = v106 + 16LL * v75;
            v77 = *(_QWORD *)v39;
            if ( v17 == *(_QWORD *)v39 )
              goto LABEL_110;
            ++v76;
          }
          if ( v74 )
            v39 = v74;
        }
        goto LABEL_110;
      }
      goto LABEL_207;
    }
LABEL_110:
    LODWORD(v107) = v69;
    if ( *(_QWORD *)v39 != -4096 )
      --HIDWORD(v107);
    *(_QWORD *)v39 = v17;
    *(_DWORD *)(v39 + 8) = 0;
    *(_DWORD *)(v39 + 8) = v29;
  }
  while ( (_DWORD)v103 );
LABEL_44:
  v43 = v106;
  v44 = 16LL * v108;
  if ( (_DWORD)v107 )
  {
    v57 = v106 + v44;
    if ( v106 != v106 + v44 )
    {
      v58 = v106;
      while ( 1 )
      {
        v59 = *(_QWORD *)v58;
        v60 = (__int64 *)v58;
        if ( *(_QWORD *)v58 != -8192 && v59 != -4096 )
          break;
        v58 += 16;
        if ( v57 == v58 )
          goto LABEL_45;
      }
      if ( v58 != v57 )
      {
        v61 = *(_DWORD *)(v58 + 8);
        v62 = &v97;
        if ( v61 == 1 )
          goto LABEL_89;
        while ( 1 )
        {
          do
          {
LABEL_82:
            v60 += 2;
            if ( v60 == (__int64 *)v57 )
              goto LABEL_86;
            while ( 1 )
            {
              v59 = *v60;
              if ( *v60 != -8192 && v59 != -4096 )
                break;
              v60 += 2;
              if ( (__int64 *)v57 == v60 )
                goto LABEL_86;
            }
            if ( v60 == (__int64 *)v57 )
            {
LABEL_86:
              v43 = v106;
              v44 = 16LL * v108;
              goto LABEL_45;
            }
            v61 = *((_DWORD *)v60 + 2);
          }
          while ( v61 != 1 );
LABEL_89:
          if ( !(_DWORD)v100 )
            break;
          v63 = (v100 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
          v64 = &v98[v63];
          v65 = *v64;
          if ( *v64 != v59 )
          {
            v66 = 1;
            v67 = 0;
            while ( v65 != -4096 )
            {
              if ( !v67 && v65 == -8192 )
                v67 = v64;
              v63 = (v100 - 1) & (v66 + v63);
              v64 = &v98[v63];
              v65 = *v64;
              if ( *v64 == v59 )
                goto LABEL_82;
              ++v66;
            }
            if ( !v67 )
              v67 = v64;
            ++v97;
            v68 = v99 + 1;
            if ( 4 * ((int)v99 + 1) < (unsigned int)(3 * v100) )
            {
              if ( (int)v100 - HIDWORD(v99) - v68 <= (unsigned int)v100 >> 3 )
              {
                v95 = v62;
                sub_2E52D10((__int64)v62, v100);
                if ( !(_DWORD)v100 )
                {
LABEL_210:
                  LODWORD(v99) = v99 + 1;
                  BUG();
                }
                v89 = (v100 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
                v62 = v95;
                v68 = v99 + 1;
                v88 = 0;
                v67 = &v98[v89];
                v90 = *v67;
                if ( *v67 != v59 )
                {
                  while ( v90 != -4096 )
                  {
                    if ( v90 == -8192 && !v88 )
                      v88 = v67;
                    v89 = (v100 - 1) & (v61 + v89);
                    v67 = &v98[v89];
                    v90 = *v67;
                    if ( *v67 == v59 )
                      goto LABEL_97;
                    ++v61;
                  }
                  goto LABEL_169;
                }
              }
              goto LABEL_97;
            }
LABEL_157:
            v94 = v62;
            sub_2E52D10((__int64)v62, 2 * v100);
            if ( !(_DWORD)v100 )
              goto LABEL_210;
            v62 = v94;
            v68 = v99 + 1;
            v86 = (v100 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
            v67 = &v98[v86];
            v87 = *v67;
            if ( *v67 != v59 )
            {
              v88 = 0;
              while ( v87 != -4096 )
              {
                if ( !v88 && v87 == -8192 )
                  v88 = v67;
                v86 = (v100 - 1) & (v61 + v86);
                v67 = &v98[v86];
                v87 = *v67;
                if ( *v67 == v59 )
                  goto LABEL_97;
                ++v61;
              }
LABEL_169:
              if ( v88 )
                v67 = v88;
            }
LABEL_97:
            LODWORD(v99) = v68;
            if ( *v67 != -4096 )
              --HIDWORD(v99);
            *v67 = v59;
            goto LABEL_82;
          }
        }
        ++v97;
        goto LABEL_157;
      }
    }
  }
LABEL_45:
  sub_C7D6A0(v43, v44, 8);
  sub_C7D6A0((__int64)v102, 8LL * (unsigned int)v104, 8);
  v45 = (__int64)v98;
  v46 = 8LL * (unsigned int)v100;
  if ( (_DWORD)v99 )
  {
    v53 = &v98[(unsigned __int64)v46 / 8];
    if ( v98 != &v98[(unsigned __int64)v46 / 8] )
    {
      v54 = v98;
      while ( *v54 == -4096 || *v54 == -8192 )
      {
        if ( v53 == ++v54 )
          return sub_C7D6A0(v45, v46, 8);
      }
      if ( v54 != v53 )
      {
        v55 = unk_501EB38;
        do
        {
          v56 = *v54++;
          *(_QWORD *)(v56 + 252) = v55;
          if ( v54 == v53 )
            break;
          while ( *v54 == -4096 || *v54 == -8192 )
          {
            if ( v53 == ++v54 )
              return sub_C7D6A0(v45, v46, 8);
          }
        }
        while ( v53 != v54 );
      }
    }
  }
  return sub_C7D6A0(v45, v46, 8);
}
