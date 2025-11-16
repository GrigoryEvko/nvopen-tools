// Function: sub_1DD1D10
// Address: 0x1dd1d10
//
__int64 __fastcall sub_1DD1D10(char *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // edi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r11
  __int64 v9; // rbx
  __int64 i; // r14
  int v11; // r13d
  char v12; // al
  unsigned int v13; // edx
  int *v14; // rax
  int v15; // ecx
  int v16; // r10d
  int *v17; // r9
  int v18; // edx
  int v19; // ebx
  int j; // r14d
  unsigned int v21; // ecx
  int v22; // esi
  unsigned int v23; // r13d
  __int64 *v24; // rdi
  unsigned int v25; // ecx
  int v26; // esi
  unsigned int v28; // ecx
  int *v29; // rdx
  int v30; // eax
  int v31; // r10d
  int v32; // edx
  int v33; // r10d
  __int64 *v34; // r10
  __int64 *v35; // rdx
  unsigned int v36; // ecx
  __int64 v37; // rax
  unsigned int v38; // r9d
  unsigned int v39; // esi
  int v40; // r13d
  unsigned int v41; // ecx
  _DWORD *v42; // rdi
  int v43; // r8d
  unsigned int v44; // r13d
  __int64 v45; // rax
  int v46; // ebx
  char *v47; // r14
  __int64 v48; // r15
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 *v51; // rax
  int v52; // r9d
  unsigned int v53; // ecx
  int v54; // r15d
  int v55; // edi
  int *v56; // rsi
  int v57; // ecx
  unsigned int v58; // r15d
  int v59; // edi
  int *v60; // rax
  int v61; // eax
  _DWORD *v62; // rdx
  int v63; // ecx
  unsigned int v64; // eax
  int v65; // r15d
  int v66; // esi
  int *v67; // rcx
  int v68; // ecx
  unsigned int v69; // r15d
  int v70; // edi
  int *v71; // rax
  unsigned int v72; // eax
  int v73; // r8d
  int v74; // edi
  _DWORD *v75; // rsi
  unsigned int v76; // eax
  int v77; // edi
  int v78; // r8d
  int v79; // r11d
  __int64 v80; // rax
  int v81; // r10d
  __int64 v82; // rdi
  int v83; // r10d
  __int64 v84; // rcx
  int v85; // r10d
  __int64 v86; // rcx
  __int64 v87; // [rsp+8h] [rbp-98h]
  __int64 v88; // [rsp+8h] [rbp-98h]
  __int64 v89; // [rsp+8h] [rbp-98h]
  __int64 v90; // [rsp+8h] [rbp-98h]
  __int64 v92; // [rsp+20h] [rbp-80h]
  char *v93; // [rsp+20h] [rbp-80h]
  unsigned int v94; // [rsp+2Ch] [rbp-74h]
  __int64 v95; // [rsp+30h] [rbp-70h] BYREF
  __int64 v96; // [rsp+38h] [rbp-68h]
  __int64 v97; // [rsp+40h] [rbp-60h]
  __int64 v98; // [rsp+48h] [rbp-58h]
  __int64 v99; // [rsp+50h] [rbp-50h] BYREF
  __int64 v100; // [rsp+58h] [rbp-48h]
  __int64 v101; // [rsp+60h] [rbp-40h]
  __int64 v102; // [rsp+68h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 48);
  v6 = *(_QWORD *)(a4 + 32);
  v95 = 0;
  v94 = v5;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  if ( v6 == a4 + 24 )
    goto LABEL_27;
  v92 = a4 + 24;
  v7 = a2;
  while ( !**(_WORD **)(v6 + 16) || **(_WORD **)(v6 + 16) == 45 )
  {
    v40 = *(_DWORD *)(*(_QWORD *)(v6 + 32) + 8LL);
    if ( !(_DWORD)v98 )
    {
      ++v95;
      goto LABEL_129;
    }
    v41 = (v98 - 1) & (37 * v40);
    v42 = (_DWORD *)(v96 + 4LL * v41);
    v43 = *v42;
    if ( v40 != *v42 )
    {
      v61 = 1;
      v62 = 0;
      while ( v43 != -1 )
      {
        if ( v43 == -2 && !v62 )
          v62 = v42;
        v79 = v61 + 1;
        v80 = ((_DWORD)v98 - 1) & (v41 + v61);
        v42 = (_DWORD *)(v96 + 4 * v80);
        v41 = v80;
        v43 = *v42;
        if ( v40 == *v42 )
          goto LABEL_67;
        v61 = v79;
      }
      if ( !v62 )
        v62 = v42;
      ++v95;
      v63 = v97 + 1;
      if ( 4 * ((int)v97 + 1) < (unsigned int)(3 * v98) )
      {
        if ( (int)v98 - HIDWORD(v97) - v63 <= (unsigned int)v98 >> 3 )
        {
          sub_136B240((__int64)&v95, v98);
          if ( !(_DWORD)v98 )
          {
LABEL_189:
            LODWORD(v97) = v97 + 1;
            BUG();
          }
          v76 = (v98 - 1) & (37 * v40);
          v75 = 0;
          v63 = v97 + 1;
          v77 = 1;
          v62 = (_DWORD *)(v96 + 4LL * v76);
          v78 = *v62;
          if ( v40 != *v62 )
          {
            while ( v78 != -1 )
            {
              if ( !v75 && v78 == -2 )
                v75 = v62;
              v76 = (v98 - 1) & (v77 + v76);
              v62 = (_DWORD *)(v96 + 4LL * v76);
              v78 = *v62;
              if ( v40 == *v62 )
                goto LABEL_111;
              ++v77;
            }
            goto LABEL_133;
          }
        }
        goto LABEL_111;
      }
LABEL_129:
      sub_136B240((__int64)&v95, 2 * v98);
      if ( !(_DWORD)v98 )
        goto LABEL_189;
      v72 = (v98 - 1) & (37 * v40);
      v63 = v97 + 1;
      v62 = (_DWORD *)(v96 + 4LL * v72);
      v73 = *v62;
      if ( v40 != *v62 )
      {
        v74 = 1;
        v75 = 0;
        while ( v73 != -1 )
        {
          if ( v73 == -2 && !v75 )
            v75 = v62;
          v72 = (v98 - 1) & (v74 + v72);
          v62 = (_DWORD *)(v96 + 4LL * v72);
          v73 = *v62;
          if ( v40 == *v62 )
            goto LABEL_111;
          ++v74;
        }
LABEL_133:
        if ( v75 )
          v62 = v75;
      }
LABEL_111:
      LODWORD(v97) = v63;
      if ( *v62 != -1 )
        --HIDWORD(v97);
      *v62 = v40;
    }
LABEL_67:
    v44 = 1;
    if ( *(_DWORD *)(v6 + 40) != 1 )
    {
      v45 = v7;
      v46 = *(_DWORD *)(v6 + 40);
      v47 = a1;
      v48 = v45;
      do
      {
        while ( 1 )
        {
          v49 = *(_QWORD *)(v6 + 32);
          if ( v48 == *(_QWORD *)(v49 + 40LL * (v44 + 1) + 24) )
            break;
          v44 += 2;
          if ( v46 == v44 )
            goto LABEL_72;
        }
        v50 = v44;
        v44 += 2;
        v51 = (__int64 *)sub_1DCC790(v47, *(_DWORD *)(v49 + 40 * v50 + 8));
        sub_1369D60(v51, v94);
      }
      while ( v46 != v44 );
LABEL_72:
      v7 = v48;
      a1 = v47;
    }
    if ( (*(_BYTE *)v6 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v6 + 46) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
    }
    v6 = *(_QWORD *)(v6 + 8);
    if ( v92 == v6 )
      goto LABEL_27;
  }
  v8 = v92;
  if ( v92 == v6 )
    goto LABEL_27;
  v93 = a1;
  while ( 2 )
  {
    v9 = *(_QWORD *)(v6 + 32);
    for ( i = v9 + 40LL * *(unsigned int *)(v6 + 40); v9 != i; v9 += 40 )
    {
      if ( *(_BYTE *)v9 )
        continue;
      v11 = *(_DWORD *)(v9 + 8);
      if ( v11 >= 0 )
        continue;
      v12 = *(_BYTE *)(v9 + 3);
      if ( (v12 & 0x10) != 0 )
      {
        if ( !(_DWORD)v98 )
        {
          ++v95;
          goto LABEL_92;
        }
        v28 = (v98 - 1) & (37 * v11);
        v29 = (int *)(v96 + 4LL * v28);
        v30 = *v29;
        if ( v11 != *v29 )
        {
          v31 = 1;
          v17 = 0;
          while ( v30 != -1 )
          {
            if ( v17 || v30 != -2 )
              v29 = v17;
            v28 = (v98 - 1) & (v31 + v28);
            v30 = *(_DWORD *)(v96 + 4LL * v28);
            if ( v11 == v30 )
              goto LABEL_9;
            ++v31;
            v17 = v29;
            v29 = (int *)(v96 + 4LL * v28);
          }
          if ( !v17 )
            v17 = v29;
          ++v95;
          v32 = v97 + 1;
          if ( 4 * ((int)v97 + 1) < (unsigned int)(3 * v98) )
          {
            if ( (int)v98 - HIDWORD(v97) - v32 <= (unsigned int)v98 >> 3 )
            {
              v88 = v8;
              sub_136B240((__int64)&v95, v98);
              if ( !(_DWORD)v98 )
              {
LABEL_188:
                LODWORD(v97) = v97 + 1;
                BUG();
              }
              v57 = 1;
              v8 = v88;
              v58 = (v98 - 1) & (37 * v11);
              v17 = (int *)(v96 + 4LL * v58);
              v59 = *v17;
              v32 = v97 + 1;
              v60 = 0;
              if ( v11 != *v17 )
              {
                while ( v59 != -1 )
                {
                  if ( v59 == -2 && !v60 )
                    v60 = v17;
                  v85 = v57 + 1;
                  v86 = ((_DWORD)v98 - 1) & (v58 + v57);
                  v17 = (int *)(v96 + 4 * v86);
                  v58 = v86;
                  v59 = *v17;
                  if ( v11 == *v17 )
                    goto LABEL_47;
                  v57 = v85;
                }
                if ( v60 )
                  v17 = v60;
              }
            }
LABEL_47:
            LODWORD(v97) = v32;
            if ( *v17 != -1 )
              --HIDWORD(v97);
LABEL_49:
            *v17 = v11;
            continue;
          }
LABEL_92:
          v87 = v8;
          sub_136B240((__int64)&v95, 2 * v98);
          if ( !(_DWORD)v98 )
            goto LABEL_188;
          v8 = v87;
          v53 = (v98 - 1) & (37 * v11);
          v17 = (int *)(v96 + 4LL * v53);
          v32 = v97 + 1;
          v54 = *v17;
          if ( v11 != *v17 )
          {
            v55 = 1;
            v56 = 0;
            while ( v54 != -1 )
            {
              if ( v54 == -2 && !v56 )
                v56 = v17;
              v81 = v55 + 1;
              v82 = ((_DWORD)v98 - 1) & (v53 + v55);
              v17 = (int *)(v96 + 4 * v82);
              v53 = v82;
              v54 = *v17;
              if ( v11 == *v17 )
                goto LABEL_47;
              v55 = v81;
            }
            if ( v56 )
              v17 = v56;
          }
          goto LABEL_47;
        }
      }
      else if ( (v12 & 0x40) != 0 )
      {
        if ( !(_DWORD)v102 )
        {
          ++v99;
          goto LABEL_115;
        }
        v13 = (v102 - 1) & (37 * v11);
        v14 = (int *)(v100 + 4LL * v13);
        v15 = *v14;
        if ( v11 != *v14 )
        {
          v16 = 1;
          v17 = 0;
          while ( v15 != -1 )
          {
            if ( v15 != -2 || v17 )
              v14 = v17;
            v13 = (v102 - 1) & (v16 + v13);
            v15 = *(_DWORD *)(v100 + 4LL * v13);
            if ( v11 == v15 )
              goto LABEL_9;
            ++v16;
            v17 = v14;
            v14 = (int *)(v100 + 4LL * v13);
          }
          if ( !v17 )
            v17 = v14;
          ++v99;
          v18 = v101 + 1;
          if ( 4 * ((int)v101 + 1) < (unsigned int)(3 * v102) )
          {
            if ( (int)v102 - HIDWORD(v101) - v18 <= (unsigned int)v102 >> 3 )
            {
              v90 = v8;
              sub_136B240((__int64)&v99, v102);
              if ( !(_DWORD)v102 )
              {
LABEL_187:
                LODWORD(v101) = v101 + 1;
                BUG();
              }
              v68 = 1;
              v8 = v90;
              v69 = (v102 - 1) & (37 * v11);
              v17 = (int *)(v100 + 4LL * v69);
              v70 = *v17;
              v18 = v101 + 1;
              v71 = 0;
              if ( v11 != *v17 )
              {
                while ( v70 != -1 )
                {
                  if ( v70 == -2 && !v71 )
                    v71 = v17;
                  v83 = v68 + 1;
                  v84 = ((_DWORD)v102 - 1) & (v69 + v68);
                  v17 = (int *)(v100 + 4 * v84);
                  v69 = v84;
                  v70 = *v17;
                  if ( v11 == *v17 )
                    goto LABEL_22;
                  v68 = v83;
                }
                if ( v71 )
                  v17 = v71;
              }
            }
            goto LABEL_22;
          }
LABEL_115:
          v89 = v8;
          sub_136B240((__int64)&v99, 2 * v102);
          if ( !(_DWORD)v102 )
            goto LABEL_187;
          v8 = v89;
          v64 = (v102 - 1) & (37 * v11);
          v17 = (int *)(v100 + 4LL * v64);
          v65 = *v17;
          v18 = v101 + 1;
          if ( v11 != *v17 )
          {
            v66 = 1;
            v67 = 0;
            while ( v65 != -1 )
            {
              if ( !v67 && v65 == -2 )
                v67 = v17;
              v64 = (v102 - 1) & (v64 + v66);
              v17 = (int *)(v100 + 4LL * v64);
              v65 = *v17;
              if ( v11 == *v17 )
                goto LABEL_22;
              ++v66;
            }
            if ( v67 )
              v17 = v67;
          }
LABEL_22:
          LODWORD(v101) = v18;
          if ( *v17 != -1 )
            --HIDWORD(v101);
          goto LABEL_49;
        }
      }
LABEL_9:
      ;
    }
    if ( (*(_BYTE *)v6 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v6 + 46) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
    }
    v6 = *(_QWORD *)(v6 + 8);
    if ( v8 != v6 )
      continue;
    break;
  }
  a1 = v93;
LABEL_27:
  v19 = *(_DWORD *)(*((_QWORD *)a1 + 44) + 32LL);
  if ( v19 )
  {
    for ( j = 0; j != v19; ++j )
    {
      while ( 1 )
      {
        v23 = j | 0x80000000;
        if ( (_DWORD)v98 )
        {
          v21 = (v98 - 1) & (37 * v23);
          v22 = *(_DWORD *)(v96 + 4LL * v21);
          if ( v23 == v22 )
            goto LABEL_30;
          v52 = 1;
          while ( v22 != -1 )
          {
            v21 = (v98 - 1) & (v52 + v21);
            v22 = *(_DWORD *)(v96 + 4LL * v21);
            if ( v23 == v22 )
              goto LABEL_30;
            ++v52;
          }
        }
        v24 = (__int64 *)sub_1DCC790(a1, v23);
        if ( (_DWORD)v102 )
          break;
LABEL_52:
        v34 = (__int64 *)v24[1];
        v35 = v24 + 1;
        if ( v34 == v24 + 1 )
          goto LABEL_30;
        v36 = *(_DWORD *)(a4 + 48);
        v37 = *v24;
        if ( v35 == (__int64 *)*v24 )
        {
          v37 = *(_QWORD *)(v37 + 8);
          v39 = v36 >> 7;
          *v24 = v37;
          v38 = *(_DWORD *)(v37 + 16);
          if ( v36 >> 7 != v38 )
          {
LABEL_55:
            if ( v39 >= v38 )
            {
              if ( v35 == (__int64 *)v37 )
              {
LABEL_90:
                *v24 = v37;
                goto LABEL_30;
              }
              while ( v39 > v38 )
              {
                v37 = *(_QWORD *)v37;
                if ( v35 == (__int64 *)v37 )
                  goto LABEL_90;
                v38 = *(_DWORD *)(v37 + 16);
              }
              goto LABEL_61;
            }
            if ( v34 == (__int64 *)v37 )
            {
              *v24 = v37;
            }
            else
            {
              do
                v37 = *(_QWORD *)(v37 + 8);
              while ( v34 != (__int64 *)v37 && *(_DWORD *)(v37 + 16) > v39 );
LABEL_61:
              *v24 = v37;
              if ( v35 == (__int64 *)v37 )
                goto LABEL_30;
            }
            if ( *(_DWORD *)(v37 + 16) != v39 )
              goto LABEL_30;
            goto LABEL_63;
          }
          if ( v35 == (__int64 *)v37 )
            goto LABEL_30;
        }
        else
        {
          v38 = *(_DWORD *)(v37 + 16);
          v39 = v36 >> 7;
          if ( v36 >> 7 != v38 )
            goto LABEL_55;
        }
LABEL_63:
        if ( (*(_QWORD *)(v37 + 8LL * ((v36 >> 6) & 1) + 24) & (1LL << v36)) != 0 )
          goto LABEL_34;
LABEL_30:
        if ( ++j == v19 )
          goto LABEL_35;
      }
      v25 = (v102 - 1) & (37 * v23);
      v26 = *(_DWORD *)(v100 + 4LL * v25);
      if ( v23 != v26 )
      {
        v33 = 1;
        while ( v26 != -1 )
        {
          v25 = (v102 - 1) & (v33 + v25);
          v26 = *(_DWORD *)(v100 + 4LL * v25);
          if ( v23 == v26 )
            goto LABEL_34;
          ++v33;
        }
        goto LABEL_52;
      }
LABEL_34:
      sub_1369D60(v24, v94);
    }
  }
LABEL_35:
  j___libc_free_0(v100);
  return j___libc_free_0(v96);
}
