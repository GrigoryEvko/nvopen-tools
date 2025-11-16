// Function: sub_1FD5CC0
// Address: 0x1fd5cc0
//
__int64 __fastcall sub_1FD5CC0(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  __int64 result; // rax
  bool v6; // cc
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 *v11; // r15
  __int64 v12; // rdi
  int v13; // r12d
  unsigned int v14; // r14d
  __int64 v15; // rdx
  int v16; // r15d
  int v17; // r13d
  __int64 v18; // r9
  unsigned int v19; // edi
  _DWORD *v20; // rax
  int v21; // ecx
  __int64 v22; // rbx
  unsigned int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // edi
  unsigned int *v26; // rcx
  unsigned int v27; // eax
  unsigned int v28; // esi
  int v29; // r12d
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // r8
  unsigned int v33; // r10d
  int v34; // edi
  int v35; // esi
  unsigned int v36; // esi
  __int64 v37; // r12
  __int64 v38; // r8
  unsigned int v39; // ecx
  __int64 *v40; // rdx
  __int64 v41; // rdi
  int v42; // r12d
  unsigned int *v43; // r11
  int v44; // eax
  int v45; // eax
  int v46; // edi
  int v47; // ecx
  int v48; // ecx
  _DWORD *v49; // r11
  __int64 v50; // r9
  int v51; // r8d
  unsigned int v52; // r10d
  int v53; // esi
  int v54; // ecx
  int v55; // ecx
  __int64 v56; // r8
  __int64 v57; // rdi
  unsigned int v58; // esi
  int v59; // r12d
  unsigned int *v60; // r10
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // r8
  __int64 v64; // rsi
  int v65; // r12d
  unsigned int v66; // edi
  __int64 *v67; // rcx
  int v68; // edi
  int v69; // r10d
  int v70; // edx
  int v71; // r11d
  __int64 *v72; // r10
  int v73; // ebx
  int v74; // edi
  int v75; // r9d
  int v76; // [rsp+8h] [rbp-68h]
  __int64 v77; // [rsp+8h] [rbp-68h]
  __int64 v78; // [rsp+10h] [rbp-60h]
  _DWORD *v79; // [rsp+10h] [rbp-60h]
  int v80; // [rsp+10h] [rbp-60h]
  __int64 v81; // [rsp+10h] [rbp-60h]
  __int64 v82; // [rsp+10h] [rbp-60h]
  __int64 *v83; // [rsp+18h] [rbp-58h]
  __int64 v86[2]; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v87[7]; // [rsp+38h] [rbp-38h] BYREF

  result = a2;
  v6 = *(_BYTE *)(a2 + 16) <= 0x17u;
  v86[0] = a2;
  if ( v6 )
  {
    v36 = *(_DWORD *)(a1 + 32);
    v37 = a1 + 8;
    if ( v36 )
    {
      v38 = *(_QWORD *)(a1 + 16);
      v39 = (v36 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v40 = (__int64 *)(v38 + 16LL * v39);
      v41 = *v40;
      if ( result == *v40 )
      {
LABEL_25:
        *((_DWORD *)v40 + 2) = a3;
        return a3;
      }
      v71 = 1;
      v72 = 0;
      while ( v41 != -8 )
      {
        if ( !v72 && v41 == -16 )
          v72 = v40;
        v39 = (v36 - 1) & (v71 + v39);
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( result == *v40 )
          goto LABEL_25;
        ++v71;
      }
      v73 = *(_DWORD *)(a1 + 24);
      if ( v72 )
        v40 = v72;
      ++*(_QWORD *)(a1 + 8);
      v74 = v73 + 1;
      if ( 4 * (v73 + 1) < 3 * v36 )
      {
        if ( v36 - *(_DWORD *)(a1 + 28) - v74 > v36 >> 3 )
        {
LABEL_83:
          *(_DWORD *)(a1 + 24) = v74;
          if ( *v40 != -8 )
            --*(_DWORD *)(a1 + 28);
          *v40 = result;
          *((_DWORD *)v40 + 2) = 0;
          goto LABEL_25;
        }
LABEL_88:
        sub_1542080(v37, v36);
        sub_154CC80(v37, v86, v87);
        v40 = (__int64 *)v87[0];
        result = v86[0];
        v74 = *(_DWORD *)(a1 + 24) + 1;
        goto LABEL_83;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    v36 *= 2;
    goto LABEL_88;
  }
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_DWORD *)(v7 + 232);
  if ( !v8 )
  {
    ++*(_QWORD *)(v7 + 208);
    goto LABEL_64;
  }
  v9 = *(_QWORD *)(v7 + 216);
  v10 = (v8 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( result != *v11 )
  {
    v69 = 1;
    v67 = 0;
    while ( v12 != -8 )
    {
      if ( !v67 && v12 == -16 )
        v67 = v11;
      v10 = (v8 - 1) & (v69 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( result == *v11 )
        goto LABEL_4;
      ++v69;
    }
    v70 = *(_DWORD *)(v7 + 224);
    if ( !v67 )
      v67 = v11;
    ++*(_QWORD *)(v7 + 208);
    v68 = v70 + 1;
    if ( 4 * (v70 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(v7 + 228) - v68 > v8 >> 3 )
        goto LABEL_66;
      goto LABEL_65;
    }
LABEL_64:
    v8 *= 2;
LABEL_65:
    sub_1542080(v7 + 208, v8);
    sub_154CC80(v7 + 208, v86, v87);
    v67 = (__int64 *)v87[0];
    result = v86[0];
    v68 = *(_DWORD *)(v7 + 224) + 1;
LABEL_66:
    *(_DWORD *)(v7 + 224) = v68;
    if ( *v67 != -8 )
      --*(_DWORD *)(v7 + 228);
    *v67 = result;
    *((_DWORD *)v67 + 2) = 0;
    goto LABEL_69;
  }
LABEL_4:
  v13 = *((_DWORD *)v11 + 2);
  if ( !v13 )
  {
    v67 = v11;
LABEL_69:
    *((_DWORD *)v67 + 2) = a3;
    return a3;
  }
  v14 = a3;
  if ( v13 == a3 )
    return result;
  if ( !a4 )
    goto LABEL_21;
  v83 = v11;
  v15 = a1;
  v16 = 0;
  v17 = 37 * a3;
  while ( 1 )
  {
    v28 = *(_DWORD *)(v7 + 528);
    v29 = v16 + v13;
    if ( v28 )
    {
      v18 = *(_QWORD *)(v7 + 512);
      v19 = (v28 - 1) & (37 * v29);
      v20 = (_DWORD *)(v18 + 8LL * v19);
      v21 = *v20;
      if ( v29 == *v20 )
        goto LABEL_9;
      v76 = 1;
      v79 = 0;
      while ( v21 != -1 )
      {
        if ( !v79 )
        {
          if ( v21 != -2 )
            v20 = 0;
          v79 = v20;
        }
        v19 = (v28 - 1) & (v76 + v19);
        v20 = (_DWORD *)(v18 + 8LL * v19);
        v21 = *v20;
        if ( v29 == *v20 )
          goto LABEL_9;
        ++v76;
      }
      if ( v79 )
        v20 = v79;
      v46 = *(_DWORD *)(v7 + 520);
      ++*(_QWORD *)(v7 + 504);
      v34 = v46 + 1;
      if ( 4 * v34 < 3 * v28 )
      {
        if ( v28 - *(_DWORD *)(v7 + 524) - v34 > v28 >> 3 )
          goto LABEL_17;
        v77 = v15;
        v80 = 37 * v29;
        sub_1392B70(v7 + 504, v28);
        v47 = *(_DWORD *)(v7 + 528);
        if ( !v47 )
        {
LABEL_128:
          ++*(_DWORD *)(v7 + 520);
          BUG();
        }
        v48 = v47 - 1;
        v49 = 0;
        v50 = *(_QWORD *)(v7 + 512);
        v34 = *(_DWORD *)(v7 + 520) + 1;
        v51 = 1;
        v15 = v77;
        v52 = v48 & v80;
        v20 = (_DWORD *)(v50 + 8LL * (v48 & (unsigned int)v80));
        v53 = *v20;
        if ( v29 == *v20 )
          goto LABEL_17;
        while ( v53 != -1 )
        {
          if ( !v49 && v53 == -2 )
            v49 = v20;
          v52 = v48 & (v51 + v52);
          v20 = (_DWORD *)(v50 + 8LL * v52);
          v53 = *v20;
          if ( v29 == *v20 )
            goto LABEL_17;
          ++v51;
        }
        goto LABEL_44;
      }
    }
    else
    {
      ++*(_QWORD *)(v7 + 504);
    }
    v78 = v15;
    sub_1392B70(v7 + 504, 2 * v28);
    v30 = *(_DWORD *)(v7 + 528);
    if ( !v30 )
      goto LABEL_128;
    v31 = v30 - 1;
    v32 = *(_QWORD *)(v7 + 512);
    v33 = v31 & (37 * v29);
    v34 = *(_DWORD *)(v7 + 520) + 1;
    v15 = v78;
    v20 = (_DWORD *)(v32 + 8LL * v33);
    v35 = *v20;
    if ( v29 == *v20 )
      goto LABEL_17;
    v75 = 1;
    v49 = 0;
    while ( v35 != -1 )
    {
      if ( v35 == -2 && !v49 )
        v49 = v20;
      v33 = v31 & (v75 + v33);
      v20 = (_DWORD *)(v32 + 8LL * v33);
      v35 = *v20;
      if ( v29 == *v20 )
        goto LABEL_17;
      ++v75;
    }
LABEL_44:
    if ( v49 )
      v20 = v49;
LABEL_17:
    *(_DWORD *)(v7 + 520) = v34;
    if ( *v20 != -1 )
      --*(_DWORD *)(v7 + 524);
    *v20 = v29;
    v20[1] = 0;
LABEL_9:
    v20[1] = v14;
    v22 = *(_QWORD *)(v15 + 40);
    v23 = *(_DWORD *)(v22 + 560);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v22 + 544);
      v25 = (v23 - 1) & v17;
      v26 = (unsigned int *)(v24 + 4LL * v25);
      v27 = *v26;
      if ( *v26 == v14 )
        goto LABEL_11;
      v42 = 1;
      v43 = 0;
      while ( v27 != -1 )
      {
        if ( v27 != -2 || v43 )
          v26 = v43;
        v25 = (v23 - 1) & (v42 + v25);
        v27 = *(_DWORD *)(v24 + 4LL * v25);
        if ( v14 == v27 )
          goto LABEL_11;
        ++v42;
        v43 = v26;
        v26 = (unsigned int *)(v24 + 4LL * v25);
      }
      v44 = *(_DWORD *)(v22 + 552);
      if ( !v43 )
        v43 = v26;
      ++*(_QWORD *)(v22 + 536);
      v45 = v44 + 1;
      if ( 4 * v45 < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(v22 + 556) - v45 > v23 >> 3 )
          goto LABEL_32;
        v82 = v15;
        sub_136B240(v22 + 536, v23);
        v61 = *(_DWORD *)(v22 + 560);
        if ( !v61 )
        {
LABEL_129:
          ++*(_DWORD *)(v22 + 552);
          BUG();
        }
        v62 = v61 - 1;
        v63 = *(_QWORD *)(v22 + 544);
        v60 = 0;
        LODWORD(v64) = v62 & v17;
        v15 = v82;
        v65 = 1;
        v43 = (unsigned int *)(v63 + 4LL * (v62 & (unsigned int)v17));
        v66 = *v43;
        v45 = *(_DWORD *)(v22 + 552) + 1;
        if ( *v43 == v14 )
          goto LABEL_32;
        while ( v66 != -1 )
        {
          if ( v66 == -2 && !v60 )
            v60 = v43;
          v64 = v62 & (unsigned int)(v64 + v65);
          v43 = (unsigned int *)(v63 + 4 * v64);
          v66 = *v43;
          if ( v14 == *v43 )
            goto LABEL_32;
          ++v65;
        }
        goto LABEL_52;
      }
    }
    else
    {
      ++*(_QWORD *)(v22 + 536);
    }
    v81 = v15;
    sub_136B240(v22 + 536, 2 * v23);
    v54 = *(_DWORD *)(v22 + 560);
    if ( !v54 )
      goto LABEL_129;
    v55 = v54 - 1;
    v56 = *(_QWORD *)(v22 + 544);
    v15 = v81;
    LODWORD(v57) = v55 & v17;
    v43 = (unsigned int *)(v56 + 4LL * (v55 & (unsigned int)v17));
    v58 = *v43;
    v45 = *(_DWORD *)(v22 + 552) + 1;
    if ( *v43 == v14 )
      goto LABEL_32;
    v59 = 1;
    v60 = 0;
    while ( v58 != -1 )
    {
      if ( !v60 && v58 == -2 )
        v60 = v43;
      v57 = v55 & (unsigned int)(v57 + v59);
      v43 = (unsigned int *)(v56 + 4 * v57);
      v58 = *v43;
      if ( v14 == *v43 )
        goto LABEL_32;
      ++v59;
    }
LABEL_52:
    if ( v60 )
      v43 = v60;
LABEL_32:
    *(_DWORD *)(v22 + 552) = v45;
    if ( *v43 != -1 )
      --*(_DWORD *)(v22 + 556);
    *v43 = v14;
LABEL_11:
    ++v16;
    ++v14;
    v17 += 37;
    if ( a4 == v16 )
      break;
    v7 = *(_QWORD *)(v15 + 40);
    v13 = *((_DWORD *)v83 + 2);
  }
  v11 = v83;
LABEL_21:
  result = a3;
  *((_DWORD *)v11 + 2) = a3;
  return result;
}
