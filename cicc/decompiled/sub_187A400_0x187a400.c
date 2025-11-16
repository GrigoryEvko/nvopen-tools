// Function: sub_187A400
// Address: 0x187a400
//
__int64 __fastcall sub_187A400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r13
  unsigned int v8; // r9d
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 *v11; // r10
  __int64 v12; // r11
  unsigned int v13; // r11d
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // r10
  __int64 *v17; // r12
  __int64 v18; // r15
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // r9d
  int v23; // r9d
  __int64 v24; // rsi
  unsigned int v25; // r10d
  __int64 *v26; // rdi
  int v27; // eax
  int v28; // edi
  int v29; // edi
  __int64 v30; // r9
  unsigned int v31; // esi
  __int64 *v32; // rcx
  __int64 v33; // r8
  int v34; // edx
  __int64 v35; // r12
  __int64 v36; // r11
  unsigned int v37; // r9d
  __int64 v38; // r8
  unsigned int v39; // edx
  __int64 *v40; // rdi
  __int64 v41; // r10
  unsigned int v42; // ecx
  __int64 v43; // rdx
  unsigned int v44; // eax
  __int64 *v45; // rdi
  __int64 v46; // r10
  __int64 *v47; // rax
  __int64 *v48; // r13
  unsigned int v49; // esi
  __int64 v50; // rax
  int v51; // esi
  int v52; // esi
  __int64 v53; // r8
  unsigned int v54; // edx
  __int64 *v55; // rcx
  int v56; // r9d
  __int64 *v57; // r10
  __int64 *v58; // rcx
  int v59; // eax
  int v60; // eax
  int v62; // edx
  int v63; // eax
  int v64; // esi
  __int64 v65; // r8
  unsigned int v66; // eax
  __int64 v67; // rdi
  int v68; // r10d
  __int64 *v69; // r9
  int v70; // edx
  int v71; // eax
  int v72; // r8d
  __int64 *v73; // r11
  int v74; // r10d
  __int64 *v75; // r11
  int v76; // [rsp+Ch] [rbp-84h]
  int v77; // [rsp+Ch] [rbp-84h]
  __int64 v80; // [rsp+20h] [rbp-70h]
  unsigned int v81; // [rsp+20h] [rbp-70h]
  int v83; // [rsp+30h] [rbp-60h]
  int v84; // [rsp+30h] [rbp-60h]
  __int64 v85; // [rsp+38h] [rbp-58h]
  __int64 *v86; // [rsp+38h] [rbp-58h]
  __int64 v87; // [rsp+38h] [rbp-58h]
  __int64 v88; // [rsp+48h] [rbp-48h] BYREF
  __int64 v89; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v90[7]; // [rsp+58h] [rbp-38h] BYREF

  v80 = (a3 - 1) / 2;
  if ( a2 < v80 )
  {
    for ( i = a2; ; i = v18 )
    {
      v19 = *(_DWORD *)(a5 + 24);
      v18 = 2 * (i + 1);
      v17 = (__int64 *)(a1 + 16 * (i + 1));
      v20 = *v17;
      v21 = *(_QWORD *)(a1 + 8 * (v18 - 1));
      v88 = *v17;
      v89 = v21;
      if ( !v19 )
        break;
      v8 = v19 - 1;
      v9 = *(_QWORD *)(a5 + 8);
      v10 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v11 = (__int64 *)(v9 + 40LL * v10);
      v12 = *v11;
      if ( v20 == *v11 )
      {
LABEL_4:
        v13 = *((_DWORD *)v11 + 2);
        goto LABEL_5;
      }
      v77 = 1;
      v26 = 0;
      while ( v12 != -4 )
      {
        if ( !v26 && v12 == -8 )
          v26 = v11;
        v10 = v8 & (v77 + v10);
        v11 = (__int64 *)(v9 + 40LL * v10);
        v12 = *v11;
        if ( v20 == *v11 )
          goto LABEL_4;
        ++v77;
      }
      v71 = *(_DWORD *)(a5 + 16);
      if ( !v26 )
        v26 = v11;
      ++*(_QWORD *)a5;
      v27 = v71 + 1;
      if ( 4 * v27 >= 3 * v19 )
        goto LABEL_12;
      if ( v19 - *(_DWORD *)(a5 + 20) - v27 <= v19 >> 3 )
      {
        sub_1874B30(a5, v19);
        sub_18721D0(a5, &v88, v90);
        v26 = (__int64 *)v90[0];
        v20 = v88;
        v27 = *(_DWORD *)(a5 + 16) + 1;
      }
LABEL_14:
      *(_DWORD *)(a5 + 16) = v27;
      if ( *v26 != -4 )
        --*(_DWORD *)(a5 + 20);
      *v26 = v20;
      *((_DWORD *)v26 + 2) = 0;
      v26[2] = 0;
      v26[3] = 0;
      v26[4] = 0;
      v19 = *(_DWORD *)(a5 + 24);
      if ( !v19 )
      {
        ++*(_QWORD *)a5;
        goto LABEL_18;
      }
      v9 = *(_QWORD *)(a5 + 8);
      v21 = v89;
      v8 = v19 - 1;
      v13 = 0;
LABEL_5:
      v14 = v8 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v15 = (__int64 *)(v9 + 40LL * v14);
      v16 = *v15;
      if ( *v15 != v21 )
      {
        v76 = 1;
        v32 = 0;
        while ( v16 != -4 )
        {
          if ( !v32 && v16 == -8 )
            v32 = v15;
          v14 = v8 & (v76 + v14);
          v15 = (__int64 *)(v9 + 40LL * v14);
          v16 = *v15;
          if ( *v15 == v21 )
            goto LABEL_6;
          ++v76;
        }
        v70 = *(_DWORD *)(a5 + 16);
        if ( !v32 )
          v32 = v15;
        ++*(_QWORD *)a5;
        v34 = v70 + 1;
        if ( 4 * v34 >= 3 * v19 )
        {
LABEL_18:
          sub_1874B30(a5, 2 * v19);
          v28 = *(_DWORD *)(a5 + 24);
          if ( !v28 )
          {
            ++*(_DWORD *)(a5 + 16);
            BUG();
          }
          v21 = v89;
          v29 = v28 - 1;
          v30 = *(_QWORD *)(a5 + 8);
          v31 = v29 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
          v32 = (__int64 *)(v30 + 40LL * v31);
          v33 = *v32;
          v34 = *(_DWORD *)(a5 + 16) + 1;
          if ( *v32 != v89 )
          {
            v74 = 1;
            v75 = 0;
            while ( v33 != -4 )
            {
              if ( v33 == -8 && !v75 )
                v75 = v32;
              v31 = v29 & (v74 + v31);
              v32 = (__int64 *)(v30 + 40LL * v31);
              v33 = *v32;
              if ( v89 == *v32 )
                goto LABEL_20;
              ++v74;
            }
            if ( v75 )
              v32 = v75;
          }
        }
        else if ( v19 - (v34 + *(_DWORD *)(a5 + 20)) <= v19 >> 3 )
        {
          sub_1874B30(a5, v19);
          sub_18721D0(a5, &v89, v90);
          v32 = (__int64 *)v90[0];
          v21 = v89;
          v34 = *(_DWORD *)(a5 + 16) + 1;
        }
LABEL_20:
        *(_DWORD *)(a5 + 16) = v34;
        if ( *v32 != -4 )
          --*(_DWORD *)(a5 + 20);
        *v32 = v21;
        *((_DWORD *)v32 + 2) = 0;
        v32[2] = 0;
        v32[3] = 0;
        v32[4] = 0;
        *(_QWORD *)(a1 + 8 * i) = *v17;
        if ( v18 >= v80 )
          goto LABEL_23;
        continue;
      }
LABEL_6:
      if ( *((_DWORD *)v15 + 2) > v13 )
        v17 = (__int64 *)(a1 + 8 * --v18);
      *(_QWORD *)(a1 + 8 * i) = *v17;
      if ( v18 >= v80 )
      {
LABEL_23:
        if ( (a3 & 1) != 0 )
          goto LABEL_24;
        goto LABEL_42;
      }
    }
    ++*(_QWORD *)a5;
LABEL_12:
    sub_1874B30(a5, 2 * v19);
    v22 = *(_DWORD *)(a5 + 24);
    if ( !v22 )
    {
      ++*(_DWORD *)(a5 + 16);
      BUG();
    }
    v23 = v22 - 1;
    v24 = *(_QWORD *)(a5 + 8);
    v25 = v23 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
    v26 = (__int64 *)(v24 + 40LL * v25);
    v20 = *v26;
    v27 = *(_DWORD *)(a5 + 16) + 1;
    if ( v88 != *v26 )
    {
      v72 = 1;
      v73 = 0;
      while ( v20 != -4 )
      {
        if ( !v73 && v20 == -8 )
          v73 = v26;
        v25 = v23 & (v72 + v25);
        v26 = (__int64 *)(v24 + 40LL * v25);
        v20 = *v26;
        if ( v88 == *v26 )
          goto LABEL_14;
        ++v72;
      }
      v20 = v88;
      if ( v73 )
        v26 = v73;
    }
    goto LABEL_14;
  }
  if ( (a3 & 1) != 0 )
  {
    v48 = (__int64 *)(a1 + 8 * a2);
    goto LABEL_54;
  }
  v18 = a2;
LABEL_42:
  if ( (a3 - 2) / 2 == v18 )
  {
    *(_QWORD *)(a1 + 8 * v18) = *(_QWORD *)(a1 + 8 * (2 * v18 + 1));
    v18 = 2 * v18 + 1;
  }
LABEL_24:
  v35 = (v18 - 1) / 2;
  if ( v18 <= a2 )
  {
    v48 = (__int64 *)(a1 + 8 * v18);
    goto LABEL_54;
  }
  v36 = a4;
  while ( 1 )
  {
    v48 = (__int64 *)(a1 + 8 * v35);
    v49 = *(_DWORD *)(a5 + 24);
    v89 = v36;
    v50 = *v48;
    v88 = *v48;
    if ( !v49 )
    {
      ++*(_QWORD *)a5;
LABEL_34:
      v85 = v36;
      sub_1874B30(a5, 2 * v49);
      v51 = *(_DWORD *)(a5 + 24);
      if ( !v51 )
      {
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a5 + 8);
      v36 = v85;
      v54 = v52 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
      v55 = (__int64 *)(v53 + 40LL * v54);
      v50 = *v55;
      if ( v88 != *v55 )
      {
        v56 = 1;
        v57 = 0;
        while ( v50 != -4 )
        {
          if ( v50 == -8 && !v57 )
            v57 = v55;
          v54 = v52 & (v56 + v54);
          v55 = (__int64 *)(v53 + 40LL * v54);
          v50 = *v55;
          if ( v88 == *v55 )
            goto LABEL_62;
          ++v56;
        }
        v50 = v88;
        if ( v57 )
          v55 = v57;
      }
      goto LABEL_62;
    }
    v37 = v49 - 1;
    v38 = *(_QWORD *)(a5 + 8);
    v39 = (v49 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v40 = (__int64 *)(v38 + 40LL * v39);
    v41 = *v40;
    if ( *v40 == v50 )
    {
LABEL_27:
      v42 = *((_DWORD *)v40 + 2);
      v43 = v36;
      goto LABEL_28;
    }
    v84 = 1;
    v55 = 0;
    while ( v41 != -4 )
    {
      if ( !v55 && v41 == -8 )
        v55 = v40;
      v39 = v37 & (v84 + v39);
      v40 = (__int64 *)(v38 + 40LL * v39);
      v41 = *v40;
      if ( v50 == *v40 )
        goto LABEL_27;
      ++v84;
    }
    v62 = *(_DWORD *)(a5 + 16);
    if ( !v55 )
      v55 = v40;
    ++*(_QWORD *)a5;
    if ( 4 * (v62 + 1) >= 3 * v49 )
      goto LABEL_34;
    if ( v49 - *(_DWORD *)(a5 + 20) - (v62 + 1) <= v49 >> 3 )
    {
      v87 = v36;
      sub_1874B30(a5, v49);
      sub_18721D0(a5, &v88, v90);
      v55 = (__int64 *)v90[0];
      v50 = v88;
      v36 = v87;
    }
LABEL_62:
    ++*(_DWORD *)(a5 + 16);
    if ( *v55 != -4 )
      --*(_DWORD *)(a5 + 20);
    *v55 = v50;
    *((_DWORD *)v55 + 2) = 0;
    v55[2] = 0;
    v55[3] = 0;
    v55[4] = 0;
    v49 = *(_DWORD *)(a5 + 24);
    if ( !v49 )
    {
      ++*(_QWORD *)a5;
      goto LABEL_66;
    }
    v38 = *(_QWORD *)(a5 + 8);
    v43 = v89;
    v37 = v49 - 1;
    v42 = 0;
LABEL_28:
    v44 = v37 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
    v45 = (__int64 *)(v38 + 40LL * v44);
    v46 = *v45;
    if ( *v45 != v43 )
      break;
LABEL_29:
    v47 = (__int64 *)(a1 + 8 * v18);
    if ( v42 >= *((_DWORD *)v45 + 2) )
    {
      v48 = (__int64 *)(a1 + 8 * v18);
      goto LABEL_54;
    }
    v18 = v35;
    *v47 = *v48;
    if ( a2 >= v35 )
      goto LABEL_54;
    v35 = (v35 - 1) / 2;
  }
  v83 = 1;
  v86 = 0;
  v81 = v42;
  while ( 1 )
  {
    v58 = v86;
    if ( v46 == -4 )
      break;
    if ( !v86 )
    {
      if ( v46 != -8 )
        v45 = 0;
      v86 = v45;
    }
    v44 = v37 & (v83 + v44);
    v45 = (__int64 *)(v38 + 40LL * v44);
    v46 = *v45;
    if ( *v45 == v43 )
    {
      v42 = v81;
      goto LABEL_29;
    }
    ++v83;
  }
  v59 = *(_DWORD *)(a5 + 16);
  if ( !v86 )
    v58 = v45;
  ++*(_QWORD *)a5;
  v60 = v59 + 1;
  if ( 4 * v60 < 3 * v49 )
  {
    if ( v49 - (*(_DWORD *)(a5 + 20) + v60) <= v49 >> 3 )
    {
      sub_1874B30(a5, v49);
      sub_18721D0(a5, &v89, v90);
      v58 = (__int64 *)v90[0];
      v43 = v89;
    }
    goto LABEL_51;
  }
LABEL_66:
  sub_1874B30(a5, 2 * v49);
  v63 = *(_DWORD *)(a5 + 24);
  if ( !v63 )
  {
    ++*(_DWORD *)(a5 + 16);
    BUG();
  }
  v43 = v89;
  v64 = v63 - 1;
  v65 = *(_QWORD *)(a5 + 8);
  v66 = (v63 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
  v58 = (__int64 *)(v65 + 40LL * v66);
  v67 = *v58;
  if ( *v58 != v89 )
  {
    v68 = 1;
    v69 = 0;
    while ( v67 != -4 )
    {
      if ( !v69 && v67 == -8 )
        v69 = v58;
      v66 = v64 & (v68 + v66);
      v58 = (__int64 *)(v65 + 40LL * v66);
      v67 = *v58;
      if ( v89 == *v58 )
        goto LABEL_51;
      ++v68;
    }
    if ( v69 )
      v58 = v69;
  }
LABEL_51:
  ++*(_DWORD *)(a5 + 16);
  if ( *v58 != -4 )
    --*(_DWORD *)(a5 + 20);
  *v58 = v43;
  v48 = (__int64 *)(a1 + 8 * v18);
  *((_DWORD *)v58 + 2) = 0;
  v58[2] = 0;
  v58[3] = 0;
  v58[4] = 0;
LABEL_54:
  *v48 = a4;
  return a4;
}
