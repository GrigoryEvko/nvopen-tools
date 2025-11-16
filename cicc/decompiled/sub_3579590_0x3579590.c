// Function: sub_3579590
// Address: 0x3579590
//
__int64 __fastcall sub_3579590(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // rcx
  int v9; // r11d
  _QWORD *v10; // r8
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // r10
  _QWORD *v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // esi
  int v17; // r10d
  _QWORD *v18; // rcx
  unsigned int v19; // r8d
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  int v24; // eax
  int v25; // edx
  unsigned int v26; // r15d
  __int64 v27; // rsi
  unsigned int v28; // r15d
  unsigned int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // rdi
  unsigned int v32; // ecx
  unsigned int v33; // r8d
  __int64 v34; // rcx
  __int64 v35; // r8
  _QWORD *v36; // rdi
  _QWORD *v37; // rsi
  __int64 v38; // r12
  unsigned int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // r9
  int v43; // r15d
  unsigned int v44; // ecx
  _QWORD *v45; // rax
  _QWORD *v46; // r11
  _QWORD *v47; // rax
  __int64 v48; // rax
  int v49; // eax
  int v50; // edx
  int v51; // eax
  int v52; // ecx
  int v53; // eax
  int v54; // ecx
  __int64 v55; // rsi
  unsigned int v56; // eax
  __int64 v57; // rdi
  int v58; // r10d
  _QWORD *v59; // r9
  int v60; // eax
  int v61; // eax
  __int64 v62; // rsi
  int v63; // r9d
  unsigned int v64; // r15d
  _QWORD *v65; // rdi
  __int64 v66; // rcx
  int v67; // eax
  int v68; // eax
  int v69; // eax
  __int64 v70; // r8
  unsigned int v71; // esi
  __int64 v72; // rdi
  int v73; // r10d
  _QWORD *v74; // r9
  int v75; // eax
  int v76; // esi
  __int64 v77; // rdi
  _QWORD *v78; // r8
  unsigned int v79; // r13d
  int v80; // r9d
  __int64 v81; // rax
  int v82; // eax
  int v83; // esi
  __int64 v84; // rdi
  _QWORD *v85; // rdx
  int v86; // r8d
  unsigned int v87; // ecx
  __int64 v88; // rax
  int v89; // eax
  int v90; // ecx
  __int64 v91; // rsi
  unsigned int v92; // r13d
  __int64 v93; // rax
  int v94; // edi
  int v95; // r8d
  __int64 v96; // [rsp+0h] [rbp-50h]
  __int64 v97; // [rsp+0h] [rbp-50h]
  unsigned int v98; // [rsp+Ch] [rbp-44h]
  unsigned int v99; // [rsp+Ch] [rbp-44h]
  _QWORD *v100; // [rsp+10h] [rbp-40h]
  _QWORD *v101; // [rsp+10h] [rbp-40h]
  __int64 v102; // [rsp+18h] [rbp-38h]
  __int64 v103; // [rsp+18h] [rbp-38h]
  _QWORD *v104; // [rsp+18h] [rbp-38h]
  _QWORD *v105; // [rsp+18h] [rbp-38h]

  v6 = a1[10];
  v7 = *(_DWORD *)(v6 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_82;
  }
  v8 = *(_QWORD *)(v6 + 8);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( a2 != (_QWORD *)*v12 )
  {
    while ( v13 != -4096 )
    {
      if ( !v10 && v13 == -8192 )
        v10 = v12;
      v11 = (v7 - 1) & (v9 + v11);
      v12 = (_QWORD *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a2 == (_QWORD *)*v12 )
        goto LABEL_3;
      ++v9;
    }
    if ( !v10 )
      v10 = v12;
    v24 = *(_DWORD *)(v6 + 16);
    ++*(_QWORD *)v6;
    v25 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(v6 + 20) - v25 <= v7 >> 3 )
      {
        sub_35793B0(v6, v7);
        v60 = *(_DWORD *)(v6 + 24);
        if ( !v60 )
          goto LABEL_155;
        v61 = v60 - 1;
        v62 = *(_QWORD *)(v6 + 8);
        v63 = 1;
        v64 = v61 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v25 = *(_DWORD *)(v6 + 16) + 1;
        v65 = 0;
        v10 = (_QWORD *)(v62 + 16LL * v64);
        v66 = *v10;
        if ( a2 != (_QWORD *)*v10 )
        {
          while ( v66 != -4096 )
          {
            if ( !v65 && v66 == -8192 )
              v65 = v10;
            v64 = v61 & (v63 + v64);
            v10 = (_QWORD *)(v62 + 16LL * v64);
            v66 = *v10;
            if ( a2 == (_QWORD *)*v10 )
              goto LABEL_20;
            ++v63;
          }
          if ( v65 )
            v10 = v65;
        }
      }
      goto LABEL_20;
    }
LABEL_82:
    sub_35793B0(v6, 2 * v7);
    v53 = *(_DWORD *)(v6 + 24);
    if ( !v53 )
      goto LABEL_155;
    v54 = v53 - 1;
    v55 = *(_QWORD *)(v6 + 8);
    v56 = (v53 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v25 = *(_DWORD *)(v6 + 16) + 1;
    v10 = (_QWORD *)(v55 + 16LL * v56);
    v57 = *v10;
    if ( a2 != (_QWORD *)*v10 )
    {
      v58 = 1;
      v59 = 0;
      while ( v57 != -4096 )
      {
        if ( v57 == -8192 && !v59 )
          v59 = v10;
        v56 = v54 & (v58 + v56);
        v10 = (_QWORD *)(v55 + 16LL * v56);
        v57 = *v10;
        if ( a2 == (_QWORD *)*v10 )
          goto LABEL_20;
        ++v58;
      }
      if ( v59 )
        v10 = v59;
    }
LABEL_20:
    *(_DWORD *)(v6 + 16) = v25;
    if ( *v10 != -4096 )
      --*(_DWORD *)(v6 + 20);
    *v10 = a2;
    v14 = 0;
    v10[1] = 0;
    goto LABEL_23;
  }
LABEL_3:
  v14 = (_QWORD *)v12[1];
  if ( a3 != v14 )
  {
    if ( a2 == v14 )
    {
      v15 = *(_QWORD *)(v6 + 8);
      v16 = *(_DWORD *)(v6 + 24);
      if ( v14 )
        goto LABEL_6;
      goto LABEL_40;
    }
LABEL_23:
    v26 = *(_DWORD *)(*a1 + 88LL);
    v27 = *(_QWORD *)(*a1 + 72LL);
    if ( v26 )
    {
      v28 = v26 - 1;
      v29 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v30;
      if ( a2 == (_QWORD *)*v30 )
      {
LABEL_25:
        v32 = *((_DWORD *)v30 + 2);
        v33 = v32 >> 6;
        v26 = v32 >> 7;
        v34 = 1LL << v32;
        v35 = v33 & 1;
      }
      else
      {
        v67 = 1;
        while ( v31 != -4096 )
        {
          v95 = v67 + 1;
          v29 = v28 & (v67 + v29);
          v30 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v30;
          if ( a2 == (_QWORD *)*v30 )
            goto LABEL_25;
          v67 = v95;
        }
        v34 = 1;
        v35 = 0;
        v26 = 0;
      }
    }
    else
    {
      v35 = 0;
      v34 = 1;
    }
    v36 = (_QWORD *)a1[5];
    v37 = a1 + 5;
    if ( a1 + 5 == v36 )
    {
      v97 = v34;
      v99 = v35;
      v101 = v14;
      v48 = sub_22077B0(0x28u);
      *(_DWORD *)(v48 + 16) = v26;
      *(_OWORD *)(v48 + 24) = 0;
      v103 = v48;
      sub_2208C80((_QWORD *)v48, (__int64)(a1 + 5));
      ++a1[7];
      v40 = v103;
      v14 = v101;
      v35 = v99;
      v34 = v97;
LABEL_39:
      a1[8] = v40;
      *(_QWORD *)(v40 + 8 * v35 + 24) |= v34;
      v6 = a1[10];
      v15 = *(_QWORD *)(v6 + 8);
      v16 = *(_DWORD *)(v6 + 24);
      if ( v14 )
      {
LABEL_6:
        if ( v16 )
        {
          v17 = 1;
          v18 = 0;
          v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v20 = (_QWORD *)(v15 + 16LL * v19);
          v21 = (_QWORD *)*v20;
          if ( a2 == (_QWORD *)*v20 )
          {
LABEL_8:
            v22 = v20 + 1;
LABEL_9:
            *v22 = a2;
            return 1;
          }
          while ( v21 != (_QWORD *)-4096LL )
          {
            if ( !v18 && v21 == (_QWORD *)-8192LL )
              v18 = v20;
            v19 = (v16 - 1) & (v17 + v19);
            v20 = (_QWORD *)(v15 + 16LL * v19);
            v21 = (_QWORD *)*v20;
            if ( a2 == (_QWORD *)*v20 )
              goto LABEL_8;
            ++v17;
          }
          if ( !v18 )
            v18 = v20;
          v49 = *(_DWORD *)(v6 + 16);
          ++*(_QWORD *)v6;
          v50 = v49 + 1;
          if ( 4 * (v49 + 1) < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(v6 + 20) - v50 > v16 >> 3 )
            {
LABEL_65:
              *(_DWORD *)(v6 + 16) = v50;
              if ( *v18 != -4096 )
                --*(_DWORD *)(v6 + 20);
              *v18 = a2;
              v22 = v18 + 1;
              v18[1] = 0;
              goto LABEL_9;
            }
            sub_35793B0(v6, v16);
            v75 = *(_DWORD *)(v6 + 24);
            if ( v75 )
            {
              v76 = v75 - 1;
              v77 = *(_QWORD *)(v6 + 8);
              v78 = 0;
              v79 = (v75 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
              v80 = 1;
              v50 = *(_DWORD *)(v6 + 16) + 1;
              v18 = (_QWORD *)(v77 + 16LL * v79);
              v81 = *v18;
              if ( a2 != (_QWORD *)*v18 )
              {
                while ( v81 != -4096 )
                {
                  if ( v81 == -8192 && !v78 )
                    v78 = v18;
                  v79 = v76 & (v80 + v79);
                  v18 = (_QWORD *)(v77 + 16LL * v79);
                  v81 = *v18;
                  if ( a2 == (_QWORD *)*v18 )
                    goto LABEL_65;
                  ++v80;
                }
                if ( v78 )
                  v18 = v78;
              }
              goto LABEL_65;
            }
LABEL_155:
            ++*(_DWORD *)(v6 + 16);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)v6;
        }
        sub_35793B0(v6, 2 * v16);
        v68 = *(_DWORD *)(v6 + 24);
        if ( v68 )
        {
          v69 = v68 - 1;
          v70 = *(_QWORD *)(v6 + 8);
          v71 = v69 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v50 = *(_DWORD *)(v6 + 16) + 1;
          v18 = (_QWORD *)(v70 + 16LL * v71);
          v72 = *v18;
          if ( a2 != (_QWORD *)*v18 )
          {
            v73 = 1;
            v74 = 0;
            while ( v72 != -4096 )
            {
              if ( v72 == -8192 && !v74 )
                v74 = v18;
              v71 = v69 & (v73 + v71);
              v18 = (_QWORD *)(v70 + 16LL * v71);
              v72 = *v18;
              if ( a2 == (_QWORD *)*v18 )
                goto LABEL_65;
              ++v73;
            }
            if ( v74 )
              v18 = v74;
          }
          goto LABEL_65;
        }
        goto LABEL_155;
      }
LABEL_40:
      if ( v16 )
      {
        v42 = 0;
        v43 = 1;
        v44 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v45 = (_QWORD *)(v15 + 16LL * v44);
        v46 = (_QWORD *)*v45;
        if ( a2 == (_QWORD *)*v45 )
        {
LABEL_42:
          v47 = v45 + 1;
LABEL_43:
          *v47 = a3;
          return 0;
        }
        while ( v46 != (_QWORD *)-4096LL )
        {
          if ( v46 == (_QWORD *)-8192LL && !v42 )
            v42 = v45;
          v44 = (v16 - 1) & (v43 + v44);
          v45 = (_QWORD *)(v15 + 16LL * v44);
          v46 = (_QWORD *)*v45;
          if ( a2 == (_QWORD *)*v45 )
            goto LABEL_42;
          ++v43;
        }
        if ( !v42 )
          v42 = v45;
        v51 = *(_DWORD *)(v6 + 16);
        ++*(_QWORD *)v6;
        v52 = v51 + 1;
        if ( 4 * (v51 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(v6 + 20) - v52 > v16 >> 3 )
          {
LABEL_78:
            *(_DWORD *)(v6 + 16) = v52;
            if ( *v42 != -4096 )
              --*(_DWORD *)(v6 + 20);
            *v42 = a2;
            v47 = v42 + 1;
            v42[1] = 0;
            goto LABEL_43;
          }
          v105 = v14;
          sub_35793B0(v6, v16);
          v89 = *(_DWORD *)(v6 + 24);
          if ( !v89 )
            goto LABEL_155;
          v90 = v89 - 1;
          v91 = *(_QWORD *)(v6 + 8);
          v85 = v105;
          v92 = (v89 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v42 = (_QWORD *)(v91 + 16LL * v92);
          v93 = *v42;
          if ( a2 != (_QWORD *)*v42 )
          {
            v94 = 1;
            while ( v93 != -4096 )
            {
              if ( !v85 && v93 == -8192 )
                v85 = v42;
              v92 = v90 & (v94 + v92);
              v42 = (_QWORD *)(v91 + 16LL * v92);
              v93 = *v42;
              if ( a2 == (_QWORD *)*v42 )
                goto LABEL_115;
              ++v94;
            }
LABEL_120:
            v52 = *(_DWORD *)(v6 + 16) + 1;
            if ( v85 )
              v42 = v85;
            goto LABEL_78;
          }
          goto LABEL_115;
        }
      }
      else
      {
        ++*(_QWORD *)v6;
      }
      v104 = v14;
      sub_35793B0(v6, 2 * v16);
      v82 = *(_DWORD *)(v6 + 24);
      if ( !v82 )
        goto LABEL_155;
      v83 = v82 - 1;
      v84 = *(_QWORD *)(v6 + 8);
      v85 = v104;
      v86 = 1;
      v87 = (v82 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v42 = (_QWORD *)(v84 + 16LL * v87);
      v88 = *v42;
      if ( a2 != (_QWORD *)*v42 )
      {
        while ( v88 != -4096 )
        {
          if ( v88 == -8192 && !v85 )
            v85 = v42;
          v87 = v83 & (v86 + v87);
          v42 = (_QWORD *)(v84 + 16LL * v87);
          v88 = *v42;
          if ( a2 == (_QWORD *)*v42 )
            goto LABEL_115;
          ++v86;
        }
        goto LABEL_120;
      }
LABEL_115:
      v52 = *(_DWORD *)(v6 + 16) + 1;
      goto LABEL_78;
    }
    v38 = a1[8];
    if ( v37 == (_QWORD *)v38 )
    {
      v38 = a1[6];
      a1[8] = v38;
      v39 = *(_DWORD *)(v38 + 16);
      if ( v39 == v26 )
        goto LABEL_34;
    }
    else
    {
      v39 = *(_DWORD *)(v38 + 16);
      if ( v39 == v26 )
        goto LABEL_35;
    }
    if ( v39 <= v26 )
    {
      if ( v37 != (_QWORD *)v38 )
      {
        while ( v26 > v39 )
        {
          v38 = *(_QWORD *)v38;
          if ( v37 == (_QWORD *)v38 )
            break;
          v39 = *(_DWORD *)(v38 + 16);
        }
      }
    }
    else if ( v36 != (_QWORD *)v38 )
    {
      do
        v38 = *(_QWORD *)(v38 + 8);
      while ( v36 != (_QWORD *)v38 && *(_DWORD *)(v38 + 16) > v26 );
    }
    a1[8] = v38;
LABEL_34:
    if ( v37 == (_QWORD *)v38 )
    {
LABEL_38:
      v96 = v34;
      v98 = v35;
      v100 = v14;
      v41 = sub_22077B0(0x28u);
      *(_DWORD *)(v41 + 16) = v26;
      *(_OWORD *)(v41 + 24) = 0;
      v102 = v41;
      sub_2208C80((_QWORD *)v41, v38);
      ++a1[7];
      v34 = v96;
      v35 = v98;
      v14 = v100;
      v40 = v102;
      goto LABEL_39;
    }
LABEL_35:
    v40 = v38;
    if ( *(_DWORD *)(v38 + 16) == v26 )
      goto LABEL_39;
    if ( *(_DWORD *)(v38 + 16) < v26 )
      v38 = *(_QWORD *)v38;
    goto LABEL_38;
  }
  return 0;
}
