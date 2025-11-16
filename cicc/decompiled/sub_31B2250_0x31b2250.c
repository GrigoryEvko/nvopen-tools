// Function: sub_31B2250
// Address: 0x31b2250
//
__int64 __fastcall sub_31B2250(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  unsigned int v8; // esi
  __int64 v10; // rdi
  int v11; // r10d
  unsigned int v12; // r15d
  unsigned int v13; // ecx
  __int64 *v14; // r13
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 result; // rax
  int v19; // r15d
  __int64 v20; // r9
  int v21; // r11d
  __int64 *v22; // rcx
  unsigned int v23; // r8d
  _QWORD *v24; // rax
  __int64 v25; // rdi
  _DWORD *v26; // rax
  bool v27; // al
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // r9
  int v32; // r11d
  __int64 *v33; // rcx
  unsigned int v34; // r8d
  __int64 *v35; // rax
  __int64 v36; // rdi
  unsigned int v37; // esi
  int v38; // esi
  int v39; // esi
  __int64 v40; // r8
  unsigned int v41; // edx
  int v42; // eax
  __int64 v43; // rdi
  int v44; // eax
  int v45; // esi
  int v46; // esi
  __int64 v47; // r8
  int v48; // r11d
  __int64 *v49; // r9
  unsigned int v50; // edx
  __int64 v51; // rdi
  int v52; // eax
  int v53; // eax
  int v54; // esi
  int v55; // esi
  __int64 v56; // r8
  unsigned int v57; // edx
  __int64 v58; // rdi
  int v59; // r10d
  __int64 *v60; // r9
  int v61; // esi
  int v62; // esi
  __int64 v63; // r8
  int v64; // r11d
  unsigned int v65; // edx
  __int64 v66; // rdi
  int v67; // ecx
  int v68; // ecx
  int v69; // eax
  int v70; // esi
  __int64 v71; // r8
  unsigned int v72; // edx
  __int64 v73; // rdi
  int v74; // r10d
  __int64 *v75; // r9
  int v76; // eax
  int v77; // edx
  __int64 v78; // rdi
  __int64 *v79; // r8
  unsigned int v80; // r15d
  int v81; // r9d
  __int64 v82; // rsi
  int v83; // r11d
  unsigned int v84; // [rsp+8h] [rbp-48h]
  unsigned int v85; // [rsp+8h] [rbp-48h]
  __int64 *v87; // [rsp+18h] [rbp-38h]

  v4 = a1 + 32;
  v8 = *(_DWORD *)(a1 + 56);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_88;
  }
  v10 = *(_QWORD *)(a1 + 40);
  v11 = 1;
  v12 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  v13 = (v8 - 1) & v12;
  v14 = (__int64 *)(v10 + 40LL * v13);
  v15 = 0;
  v16 = *v14;
  if ( *v14 == a4 )
  {
LABEL_3:
    v17 = (__int64)(v14 + 1);
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v15 )
      v15 = v14;
    v13 = (v8 - 1) & (v11 + v13);
    v14 = (__int64 *)(v10 + 40LL * v13);
    v16 = *v14;
    if ( *v14 == a4 )
      goto LABEL_3;
    ++v11;
  }
  v67 = *(_DWORD *)(a1 + 48);
  if ( !v15 )
    v15 = v14;
  ++*(_QWORD *)(a1 + 32);
  v68 = v67 + 1;
  if ( 4 * v68 >= 3 * v8 )
  {
LABEL_88:
    sub_31B1C10(v4, 2 * v8);
    v69 = *(_DWORD *)(a1 + 56);
    if ( v69 )
    {
      v70 = v69 - 1;
      v71 = *(_QWORD *)(a1 + 40);
      v68 = *(_DWORD *)(a1 + 48) + 1;
      v72 = (v69 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v15 = (__int64 *)(v71 + 40LL * v72);
      v73 = *v15;
      if ( *v15 != a4 )
      {
        v74 = 1;
        v75 = 0;
        while ( v73 != -4096 )
        {
          if ( !v75 && v73 == -8192 )
            v75 = v15;
          v72 = v70 & (v74 + v72);
          v15 = (__int64 *)(v71 + 40LL * v72);
          v73 = *v15;
          if ( *v15 == a4 )
            goto LABEL_83;
          ++v74;
        }
        if ( v75 )
          v15 = v75;
      }
      goto LABEL_83;
    }
    goto LABEL_131;
  }
  if ( v8 - *(_DWORD *)(a1 + 52) - v68 <= v8 >> 3 )
  {
    sub_31B1C10(v4, v8);
    v76 = *(_DWORD *)(a1 + 56);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = *(_QWORD *)(a1 + 40);
      v79 = 0;
      v80 = (v76 - 1) & v12;
      v81 = 1;
      v68 = *(_DWORD *)(a1 + 48) + 1;
      v15 = (__int64 *)(v78 + 40LL * v80);
      v82 = *v15;
      if ( *v15 != a4 )
      {
        while ( v82 != -4096 )
        {
          if ( !v79 && v82 == -8192 )
            v79 = v15;
          v80 = v77 & (v81 + v80);
          v15 = (__int64 *)(v78 + 40LL * v80);
          v82 = *v15;
          if ( *v15 == a4 )
            goto LABEL_83;
          ++v81;
        }
        if ( v79 )
          v15 = v79;
      }
      goto LABEL_83;
    }
LABEL_131:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_83:
  *(_DWORD *)(a1 + 48) = v68;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 52);
  v15[1] = 0;
  v17 = (__int64)(v15 + 1);
  v15[2] = 0;
  *v15 = a4;
  v15[3] = 0;
  *((_DWORD *)v15 + 8) = 0;
LABEL_4:
  result = (__int64)&a2[a3];
  v19 = 0;
  v87 = (__int64 *)result;
  while ( v87 != a2 )
  {
    v30 = *(_DWORD *)(a1 + 24);
    v28 = *a2;
    if ( v30 )
    {
      v31 = *(_QWORD *)(a1 + 8);
      v32 = 1;
      v33 = 0;
      v34 = (v30 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v35 = (__int64 *)(v31 + 16LL * v34);
      v36 = *v35;
      if ( v28 == *v35 )
        goto LABEL_19;
      while ( v36 != -4096 )
      {
        if ( v33 || v36 != -8192 )
          v35 = v33;
        v34 = (v30 - 1) & (v32 + v34);
        v36 = *(_QWORD *)(v31 + 16LL * v34);
        if ( v28 == v36 )
          goto LABEL_19;
        ++v32;
        v33 = v35;
        v35 = (__int64 *)(v31 + 16LL * v34);
      }
      if ( !v33 )
        v33 = v35;
      v52 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v53 = v52 + 1;
      if ( 4 * v53 < 3 * v30 )
      {
        if ( v30 - *(_DWORD *)(a1 + 20) - v53 > v30 >> 3 )
          goto LABEL_51;
        v85 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
        sub_31B1E90(a1, v30);
        v61 = *(_DWORD *)(a1 + 24);
        if ( !v61 )
        {
LABEL_129:
          ++*(_DWORD *)(a1 + 16);
          BUG();
        }
        v62 = v61 - 1;
        v63 = *(_QWORD *)(a1 + 8);
        v64 = 1;
        v60 = 0;
        v65 = v62 & v85;
        v53 = *(_DWORD *)(a1 + 16) + 1;
        v33 = (__int64 *)(v63 + 16LL * (v62 & v85));
        v66 = *v33;
        if ( v28 == *v33 )
          goto LABEL_51;
        while ( v66 != -4096 )
        {
          if ( v66 == -8192 && !v60 )
            v60 = v33;
          v65 = v62 & (v64 + v65);
          v33 = (__int64 *)(v63 + 16LL * v65);
          v66 = *v33;
          if ( v28 == *v33 )
            goto LABEL_51;
          ++v64;
        }
        goto LABEL_62;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_31B1E90(a1, 2 * v30);
    v54 = *(_DWORD *)(a1 + 24);
    if ( !v54 )
      goto LABEL_129;
    v55 = v54 - 1;
    v56 = *(_QWORD *)(a1 + 8);
    v57 = v55 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v53 = *(_DWORD *)(a1 + 16) + 1;
    v33 = (__int64 *)(v56 + 16LL * v57);
    v58 = *v33;
    if ( v28 == *v33 )
      goto LABEL_51;
    v59 = 1;
    v60 = 0;
    while ( v58 != -4096 )
    {
      if ( v58 != -8192 || v60 )
        v33 = v60;
      v57 = v55 & (v59 + v57);
      v58 = *(_QWORD *)(v56 + 16LL * v57);
      if ( v28 == v58 )
      {
        v33 = (__int64 *)(v56 + 16LL * v57);
        goto LABEL_51;
      }
      ++v59;
      v60 = v33;
      v33 = (__int64 *)(v56 + 16LL * v57);
    }
LABEL_62:
    if ( v60 )
      v33 = v60;
LABEL_51:
    *(_DWORD *)(a1 + 16) = v53;
    if ( *v33 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *v33 = v28;
    v33[1] = a4;
LABEL_19:
    v37 = *(_DWORD *)(v17 + 24);
    if ( !v37 )
    {
      ++*(_QWORD *)v17;
      goto LABEL_21;
    }
    v20 = *(_QWORD *)(v17 + 8);
    v21 = 1;
    v22 = 0;
    v23 = (v37 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v24 = (_QWORD *)(v20 + 16LL * v23);
    v25 = *v24;
    if ( v28 != *v24 )
    {
      while ( v25 != -4096 )
      {
        if ( !v22 && v25 == -8192 )
          v22 = v24;
        v23 = (v37 - 1) & (v21 + v23);
        v24 = (_QWORD *)(v20 + 16LL * v23);
        v25 = *v24;
        if ( v28 == *v24 )
          goto LABEL_7;
        ++v21;
      }
      if ( !v22 )
        v22 = v24;
      v44 = *(_DWORD *)(v17 + 16);
      ++*(_QWORD *)v17;
      v42 = v44 + 1;
      if ( 4 * v42 >= 3 * v37 )
      {
LABEL_21:
        sub_31B2070(v17, 2 * v37);
        v38 = *(_DWORD *)(v17 + 24);
        if ( !v38 )
          goto LABEL_130;
        v39 = v38 - 1;
        v40 = *(_QWORD *)(v17 + 8);
        v41 = v39 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v42 = *(_DWORD *)(v17 + 16) + 1;
        v22 = (__int64 *)(v40 + 16LL * v41);
        v43 = *v22;
        if ( v28 != *v22 )
        {
          v83 = 1;
          v49 = 0;
          while ( v43 != -4096 )
          {
            if ( v43 == -8192 && !v49 )
              v49 = v22;
            v41 = v39 & (v83 + v41);
            v22 = (__int64 *)(v40 + 16LL * v41);
            v43 = *v22;
            if ( v28 == *v22 )
              goto LABEL_23;
            ++v83;
          }
          goto LABEL_39;
        }
      }
      else if ( v37 - *(_DWORD *)(v17 + 20) - v42 <= v37 >> 3 )
      {
        v84 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
        sub_31B2070(v17, v37);
        v45 = *(_DWORD *)(v17 + 24);
        if ( !v45 )
        {
LABEL_130:
          ++*(_DWORD *)(v17 + 16);
          BUG();
        }
        v46 = v45 - 1;
        v47 = *(_QWORD *)(v17 + 8);
        v48 = 1;
        v49 = 0;
        v50 = v46 & v84;
        v42 = *(_DWORD *)(v17 + 16) + 1;
        v22 = (__int64 *)(v47 + 16LL * (v46 & v84));
        v51 = *v22;
        if ( v28 != *v22 )
        {
          while ( v51 != -4096 )
          {
            if ( v51 == -8192 && !v49 )
              v49 = v22;
            v50 = v46 & (v48 + v50);
            v22 = (__int64 *)(v47 + 16LL * v50);
            v51 = *v22;
            if ( v28 == *v22 )
              goto LABEL_23;
            ++v48;
          }
LABEL_39:
          if ( v49 )
            v22 = v49;
        }
      }
LABEL_23:
      *(_DWORD *)(v17 + 16) = v42;
      if ( *v22 != -4096 )
        --*(_DWORD *)(v17 + 20);
      *v22 = v28;
      v26 = v22 + 1;
      *((_DWORD *)v22 + 2) = 0;
      goto LABEL_8;
    }
LABEL_7:
    v26 = v24 + 1;
LABEL_8:
    *v26 = v19;
    v27 = sub_318B630(v28);
    if ( v28 && v27 && (*(_DWORD *)(v28 + 8) != 37 || sub_318B6C0(v28)) )
    {
      if ( sub_318B670(v28) )
      {
        v28 = sub_318B680(v28);
      }
      else if ( *(_DWORD *)(v28 + 8) == 37 )
      {
        v28 = sub_318B6C0(v28);
      }
    }
    v29 = *sub_318EB80(v28);
    result = 1;
    if ( *(_BYTE *)(v29 + 8) == 17 )
      result = *(unsigned int *)(v29 + 32);
    v19 += result;
    ++a2;
  }
  return result;
}
