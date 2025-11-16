// Function: sub_11F7FC0
// Address: 0x11f7fc0
//
__int64 __fastcall sub_11F7FC0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // r15
  _QWORD *v7; // r14
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rax
  double v20; // xmm0_8
  double v21; // xmm1_8
  __int64 result; // rax
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // r14
  unsigned int v26; // r8d
  unsigned int v27; // r13d
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // r9
  int v31; // eax
  int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // edx
  int v35; // eax
  __int64 *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdi
  __int64 v39; // r15
  int v40; // r10d
  int v41; // r10d
  int v42; // eax
  int v43; // eax
  int v44; // ecx
  __int64 v45; // r9
  int v46; // r8d
  unsigned int v47; // r13d
  __int64 *v48; // rsi
  __int64 v49; // rdx
  int v50; // r10d
  __int64 *v51; // r8

  v6 = sub_C52410();
  v7 = v6 + 1;
  v8 = sub_C959E0();
  v9 = (_QWORD *)v6[2];
  if ( v9 )
  {
    v10 = v6 + 1;
    do
    {
      while ( 1 )
      {
        v11 = v9[2];
        v12 = v9[3];
        if ( v8 <= v9[4] )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v10 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v11 );
LABEL_6:
    if ( v7 != v10 && v8 >= v10[4] )
      v7 = v10;
  }
  if ( v7 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_23;
  v13 = v7[7];
  if ( !v13 )
    goto LABEL_23;
  v14 = v7 + 6;
  do
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v13 + 16);
      v16 = *(_QWORD *)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) >= dword_4F91EE8 )
        break;
      v13 = *(_QWORD *)(v13 + 24);
      if ( !v16 )
        goto LABEL_15;
    }
    v14 = (_QWORD *)v13;
    v13 = *(_QWORD *)(v13 + 16);
  }
  while ( v15 );
LABEL_15:
  if ( v7 + 6 == v14 )
    goto LABEL_23;
  if ( dword_4F91EE8 < *((_DWORD *)v14 + 8) )
    goto LABEL_23;
  if ( *((int *)v14 + 9) <= 0 )
    goto LABEL_23;
  v17 = *(_QWORD *)(a3 + 8);
  if ( !v17 )
    goto LABEL_23;
  v18 = sub_FDD860(*(__int64 **)(a3 + 8), a2);
  v19 = sub_FDC4B0(v17);
  if ( v18 < 0 )
  {
    v20 = (double)(int)(v18 & 1 | ((unsigned __int64)v18 >> 1)) + (double)(int)(v18 & 1 | ((unsigned __int64)v18 >> 1));
    if ( v19 >= 0 )
      goto LABEL_21;
  }
  else
  {
    v20 = (double)(int)v18;
    if ( v19 >= 0 )
    {
LABEL_21:
      v21 = (double)(int)v19;
      goto LABEL_22;
    }
  }
  v21 = (double)(int)(v19 & 1 | ((unsigned __int64)v19 >> 1)) + (double)(int)(v19 & 1 | ((unsigned __int64)v19 >> 1));
LABEL_22:
  result = 1;
  if ( *(double *)&qword_4F91F68 > v20 / v21 )
    return result;
LABEL_23:
  if ( (_BYTE)qword_4F92128 )
  {
    v23 = *(_DWORD *)(a1 + 32);
    v24 = *(_QWORD *)(a1 + 16);
    v25 = a1 + 8;
    if ( !v23 )
      goto LABEL_30;
    goto LABEL_25;
  }
  result = (unsigned __int8)qword_4F92048;
  if ( !(_BYTE)qword_4F92048 )
    return result;
  v23 = *(_DWORD *)(a1 + 32);
  v24 = *(_QWORD *)(a1 + 16);
  v25 = a1 + 8;
  if ( v23 )
  {
LABEL_25:
    v26 = v23 - 1;
    v27 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v28 = (v23 - 1) & v27;
    v29 = (__int64 *)(v24 + 16LL * v28);
    v30 = *v29;
    if ( *v29 == a2 )
      return *((unsigned __int8 *)v29 + 8);
    v38 = *v29;
    LODWORD(v39) = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v40 = 1;
    while ( v38 != -4096 )
    {
      v39 = v26 & ((_DWORD)v39 + v40);
      v38 = *(_QWORD *)(v24 + 16 * v39);
      if ( v38 == a2 )
        goto LABEL_42;
      ++v40;
    }
  }
LABEL_30:
  sub_11F6970(a1, *(_QWORD *)(a2 + 72));
  v23 = *(_DWORD *)(a1 + 32);
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_32;
  }
  v26 = v23 - 1;
  v24 = *(_QWORD *)(a1 + 16);
  v27 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v28 = (v23 - 1) & v27;
  v29 = (__int64 *)(v24 + 16LL * v28);
  v30 = *v29;
  if ( *v29 == a2 )
    return *((unsigned __int8 *)v29 + 8);
LABEL_42:
  v41 = 1;
  v36 = 0;
  while ( v30 != -4096 )
  {
    if ( v30 == -8192 && !v36 )
      v36 = v29;
    v28 = v26 & (v41 + v28);
    v29 = (__int64 *)(v24 + 16LL * v28);
    v30 = *v29;
    if ( *v29 == a2 )
      return *((unsigned __int8 *)v29 + 8);
    ++v41;
  }
  if ( !v36 )
    v36 = v29;
  v42 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v35 = v42 + 1;
  if ( 4 * v35 >= 3 * v23 )
  {
LABEL_32:
    sub_11F63E0(v25, 2 * v23);
    v31 = *(_DWORD *)(a1 + 32);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 16);
      v34 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v35 = *(_DWORD *)(a1 + 24) + 1;
      v36 = (__int64 *)(v33 + 16LL * v34);
      v37 = *v36;
      if ( *v36 != a2 )
      {
        v50 = 1;
        v51 = 0;
        while ( v37 != -4096 )
        {
          if ( !v51 && v37 == -8192 )
            v51 = v36;
          v34 = v32 & (v50 + v34);
          v36 = (__int64 *)(v33 + 16LL * v34);
          v37 = *v36;
          if ( *v36 == a2 )
            goto LABEL_34;
          ++v50;
        }
        if ( v51 )
          v36 = v51;
      }
      goto LABEL_34;
    }
LABEL_77:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  if ( v23 - (v35 + *(_DWORD *)(a1 + 28)) <= v23 >> 3 )
  {
    sub_11F63E0(v25, v23);
    v43 = *(_DWORD *)(a1 + 32);
    if ( !v43 )
      goto LABEL_77;
    v44 = v43 - 1;
    v45 = *(_QWORD *)(a1 + 16);
    v46 = 1;
    v47 = (v43 - 1) & v27;
    v48 = 0;
    v35 = *(_DWORD *)(a1 + 24) + 1;
    v36 = (__int64 *)(v45 + 16LL * v47);
    v49 = *v36;
    if ( *v36 != a2 )
    {
      while ( v49 != -4096 )
      {
        if ( v49 == -8192 && !v48 )
          v48 = v36;
        v47 = v44 & (v46 + v47);
        v36 = (__int64 *)(v45 + 16LL * v47);
        v49 = *v36;
        if ( *v36 == a2 )
          goto LABEL_34;
        ++v46;
      }
      if ( v48 )
        v36 = v48;
    }
  }
LABEL_34:
  *(_DWORD *)(a1 + 24) = v35;
  if ( *v36 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v36 = a2;
  *((_BYTE *)v36 + 8) = 0;
  return 0;
}
