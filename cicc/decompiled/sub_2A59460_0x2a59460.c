// Function: sub_2A59460
// Address: 0x2a59460
//
__int64 __fastcall sub_2A59460(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // r9d
  __int64 v10; // r11
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v14; // r10
  __int64 v15; // r15
  int v16; // r13d
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // r12
  _QWORD *v23; // rax
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // edx
  int v28; // eax
  __int64 *v29; // r8
  __int64 v30; // rsi
  int v31; // r11d
  __int64 *v32; // r9
  __int64 v33; // rax
  unsigned int v34; // esi
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 *v37; // r10
  int v38; // r11d
  unsigned int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // rdx
  int v42; // ecx
  int v43; // eax
  int v44; // eax
  int v45; // eax
  __int64 v46; // r8
  unsigned int v47; // ecx
  int v48; // edx
  int v49; // eax
  int v50; // eax
  int v51; // esi
  __int64 v52; // r9
  __int64 *v53; // r11
  unsigned int v54; // ecx
  int v55; // eax
  int v56; // eax
  int v57; // ecx
  __int64 v58; // rdi
  int v59; // r11d
  unsigned int v60; // edx
  __int64 v61; // rsi
  int v62; // r11d
  __int64 *v63; // r9
  __int64 v65[7]; // [rsp+8h] [rbp-38h] BYREF

  v65[0] = a2;
  v7 = *(_DWORD *)(a3 + 24);
  v8 = *(_QWORD *)(a3 + 8);
  if ( v7 )
  {
    v9 = v7 - 1;
    LODWORD(v10) = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v8 + 16LL * (unsigned int)v10);
    v12 = *v11;
    if ( a2 == *v11 )
      return v11[1];
    v14 = *v11;
    LODWORD(v15) = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = 1;
    while ( v14 != -4096 )
    {
      v15 = v9 & ((_DWORD)v15 + v16);
      v14 = *(_QWORD *)(v8 + 16 * v15);
      if ( a2 == v14 )
        goto LABEL_29;
      ++v16;
    }
  }
  if ( a2 )
  {
    v17 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v18 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  if ( v18 < *(_DWORD *)(a4 + 32) )
  {
    if ( *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v17) )
    {
      sub_102DBD0(a1 + 432, a2);
      if ( v19 )
      {
        if ( v65[0] )
        {
          v20 = (unsigned int)(*(_DWORD *)(v65[0] + 44) + 1);
          v21 = *(_DWORD *)(v65[0] + 44) + 1;
        }
        else
        {
          v20 = 0;
          v21 = 0;
        }
        if ( v21 >= *(_DWORD *)(a4 + 32) )
          BUG();
        v22 = sub_2A59460(a1, **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v20) + 8LL), a3);
        v23 = sub_11D31A0(a3, v65);
        goto LABEL_15;
      }
    }
  }
  v33 = sub_ACA8A0(*(__int64 ***)(a3 + 96));
  v34 = *(_DWORD *)(a3 + 24);
  v22 = v33;
  if ( !v34 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_39;
  }
  v35 = v65[0];
  v36 = *(_QWORD *)(a3 + 8);
  v37 = 0;
  v38 = 1;
  v39 = (v34 - 1) & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
  v40 = (__int64 *)(v36 + 16LL * v39);
  v41 = *v40;
  if ( v65[0] == *v40 )
  {
LABEL_26:
    v23 = v40 + 1;
    goto LABEL_15;
  }
  while ( v41 != -4096 )
  {
    if ( !v37 && v41 == -8192 )
      v37 = v40;
    v39 = (v34 - 1) & (v38 + v39);
    v40 = (__int64 *)(v36 + 16LL * v39);
    v41 = *v40;
    if ( v65[0] == *v40 )
      goto LABEL_26;
    ++v38;
  }
  if ( !v37 )
    v37 = v40;
  v49 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v48 = v49 + 1;
  if ( 4 * (v49 + 1) >= 3 * v34 )
  {
LABEL_39:
    sub_116E750(a3, 2 * v34);
    v44 = *(_DWORD *)(a3 + 24);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a3 + 8);
      v47 = v45 & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
      v48 = *(_DWORD *)(a3 + 16) + 1;
      v37 = (__int64 *)(v46 + 16LL * v47);
      v35 = *v37;
      if ( v65[0] != *v37 )
      {
        v62 = 1;
        v63 = 0;
        while ( v35 != -4096 )
        {
          if ( !v63 && v35 == -8192 )
            v63 = v37;
          v47 = v45 & (v62 + v47);
          v37 = (__int64 *)(v46 + 16LL * v47);
          v35 = *v37;
          if ( v65[0] == *v37 )
            goto LABEL_41;
          ++v62;
        }
        v35 = v65[0];
        if ( v63 )
          v37 = v63;
      }
      goto LABEL_41;
    }
    goto LABEL_98;
  }
  if ( v34 - *(_DWORD *)(a3 + 20) - v48 <= v34 >> 3 )
  {
    sub_116E750(a3, v34);
    v50 = *(_DWORD *)(a3 + 24);
    if ( v50 )
    {
      v51 = v50 - 1;
      v52 = *(_QWORD *)(a3 + 8);
      v53 = 0;
      v54 = (v50 - 1) & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
      v37 = (__int64 *)(v52 + 16LL * v54);
      v35 = *v37;
      v48 = *(_DWORD *)(a3 + 16) + 1;
      v55 = 1;
      if ( v65[0] != *v37 )
      {
        while ( v35 != -4096 )
        {
          if ( !v53 && v35 == -8192 )
            v53 = v37;
          v54 = v51 & (v55 + v54);
          v37 = (__int64 *)(v52 + 16LL * v54);
          v35 = *v37;
          if ( v65[0] == *v37 )
            goto LABEL_41;
          ++v55;
        }
        v35 = v65[0];
        if ( v53 )
          v37 = v53;
      }
      goto LABEL_41;
    }
LABEL_98:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_41:
  *(_DWORD *)(a3 + 16) = v48;
  if ( *v37 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v37 = v35;
  v23 = v37 + 1;
  v37[1] = 0;
LABEL_15:
  *v23 = v22;
  v7 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_17;
  }
  v14 = v65[0];
  v9 = v7 - 1;
  v8 = *(_QWORD *)(a3 + 8);
  v10 = (v7 - 1) & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
  v11 = (__int64 *)(v8 + 16 * v10);
  v12 = *v11;
  if ( *v11 == v65[0] )
    return v11[1];
LABEL_29:
  v42 = 1;
  v29 = 0;
  while ( v12 != -4096 )
  {
    if ( !v29 && v12 == -8192 )
      v29 = v11;
    LODWORD(v10) = v9 & (v42 + v10);
    v11 = (__int64 *)(v8 + 16LL * (unsigned int)v10);
    v12 = *v11;
    if ( *v11 == v14 )
      return v11[1];
    ++v42;
  }
  if ( !v29 )
    v29 = v11;
  v43 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v28 = v43 + 1;
  if ( 4 * v28 >= 3 * v7 )
  {
LABEL_17:
    sub_116E750(a3, 2 * v7);
    v24 = *(_DWORD *)(a3 + 24);
    if ( v24 )
    {
      v14 = v65[0];
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a3 + 8);
      v27 = (v24 - 1) & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
      v28 = *(_DWORD *)(a3 + 16) + 1;
      v29 = (__int64 *)(v26 + 16LL * v27);
      v30 = *v29;
      if ( *v29 != v65[0] )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( !v32 && v30 == -8192 )
            v32 = v29;
          v27 = v25 & (v31 + v27);
          v29 = (__int64 *)(v26 + 16LL * v27);
          v30 = *v29;
          if ( v65[0] == *v29 )
            goto LABEL_35;
          ++v31;
        }
LABEL_21:
        if ( v32 )
          v29 = v32;
        goto LABEL_35;
      }
      goto LABEL_35;
    }
LABEL_96:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
  if ( v7 - (v28 + *(_DWORD *)(a3 + 20)) > v7 >> 3 )
    goto LABEL_35;
  sub_116E750(a3, v7);
  v56 = *(_DWORD *)(a3 + 24);
  if ( !v56 )
    goto LABEL_96;
  v14 = v65[0];
  v57 = v56 - 1;
  v58 = *(_QWORD *)(a3 + 8);
  v32 = 0;
  v59 = 1;
  v60 = (v56 - 1) & ((LODWORD(v65[0]) >> 9) ^ (LODWORD(v65[0]) >> 4));
  v28 = *(_DWORD *)(a3 + 16) + 1;
  v29 = (__int64 *)(v58 + 16LL * v60);
  v61 = *v29;
  if ( *v29 != v65[0] )
  {
    while ( v61 != -4096 )
    {
      if ( !v32 && v61 == -8192 )
        v32 = v29;
      v60 = v57 & (v59 + v60);
      v29 = (__int64 *)(v58 + 16LL * v60);
      v61 = *v29;
      if ( v65[0] == *v29 )
        goto LABEL_35;
      ++v59;
    }
    goto LABEL_21;
  }
LABEL_35:
  *(_DWORD *)(a3 + 16) = v28;
  if ( *v29 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v29 = v14;
  v29[1] = 0;
  return 0;
}
