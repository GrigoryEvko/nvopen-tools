// Function: sub_2120330
// Address: 0x2120330
//
unsigned __int64 __fastcall sub_2120330(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v5; // eax
  char v6; // cl
  __int64 v7; // rdx
  int v8; // esi
  int v9; // r9d
  unsigned int v10; // edi
  int *v11; // r14
  int v12; // r8d
  __int64 v13; // rax
  char v14; // al
  unsigned int v15; // esi
  __int64 v16; // r10
  int v17; // esi
  int v18; // edx
  unsigned int v19; // ecx
  __int64 v20; // rdi
  int v21; // r9d
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned int v25; // edi
  __int64 v26; // r8
  int v27; // edx
  unsigned int v28; // ecx
  int v29; // eax
  int v30; // r11d
  __int64 v31; // rdi
  int v32; // eax
  int v33; // edx
  unsigned int v34; // ecx
  int v35; // esi
  __int64 v36; // rdi
  int v37; // edx
  int v38; // eax
  unsigned int v39; // esi
  int v40; // ecx
  int v41; // r10d
  __int64 v42; // r9
  int v43; // eax
  int v44; // edx
  __int64 v45; // r12
  int v46; // r10d

  v5 = sub_200F8F0(a1, a2, a3);
  v6 = *(_BYTE *)(a1 + 752) & 1;
  if ( v6 )
  {
    v7 = a1 + 760;
    v8 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 768);
    v7 = *(_QWORD *)(a1 + 760);
    if ( !(_DWORD)v23 )
      goto LABEL_16;
    v8 = v23 - 1;
  }
  v9 = 1;
  v10 = v8 & (37 * v5);
  v11 = (int *)(v7 + 8LL * v10);
  v12 = *v11;
  if ( v5 == *v11 )
    goto LABEL_4;
  while ( v12 != -1 )
  {
    v10 = v8 & (v9 + v10);
    v11 = (int *)(v7 + 8LL * v10);
    v12 = *v11;
    if ( v5 == *v11 )
      goto LABEL_4;
    ++v9;
  }
  if ( v6 )
  {
    v24 = 64;
    goto LABEL_17;
  }
  v23 = *(unsigned int *)(a1 + 768);
LABEL_16:
  v24 = 8 * v23;
LABEL_17:
  v11 = (int *)(v7 + v24);
LABEL_4:
  v13 = 64;
  if ( !v6 )
    v13 = 8LL * *(unsigned int *)(a1 + 768);
  if ( v11 == (int *)(v13 + v7) )
    return a2;
  sub_200D1B0(a1, v11 + 1);
  v14 = *(_BYTE *)(a1 + 352) & 1;
  if ( v14 )
  {
    v16 = a1 + 360;
    v17 = 7;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 368);
    v16 = *(_QWORD *)(a1 + 360);
    if ( !v15 )
    {
      v25 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v26 = 0;
      v27 = (v25 >> 1) + 1;
LABEL_25:
      v28 = 3 * v15;
      goto LABEL_26;
    }
    v17 = v15 - 1;
  }
  v18 = v11[1];
  v19 = v17 & (37 * v18);
  v20 = v16 + 24LL * v19;
  v21 = *(_DWORD *)v20;
  if ( v18 == *(_DWORD *)v20 )
    return *(_QWORD *)(v20 + 8);
  v30 = 1;
  v26 = 0;
  while ( v21 != -1 )
  {
    if ( v26 || v21 != -2 )
      v20 = v26;
    v19 = v17 & (v30 + v19);
    v45 = v16 + 24LL * v19;
    v21 = *(_DWORD *)v45;
    if ( v18 == *(_DWORD *)v45 )
      return *(_QWORD *)(v45 + 8);
    v26 = v20;
    ++v30;
    v20 = v16 + 24LL * v19;
  }
  v28 = 24;
  v15 = 8;
  if ( !v26 )
    v26 = v20;
  v25 = *(_DWORD *)(a1 + 352);
  ++*(_QWORD *)(a1 + 344);
  v27 = (v25 >> 1) + 1;
  if ( !v14 )
  {
    v15 = *(_DWORD *)(a1 + 368);
    goto LABEL_25;
  }
LABEL_26:
  if ( v28 <= 4 * v27 )
  {
    sub_200F500(a1 + 344, 2 * v15);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v31 = a1 + 360;
      v32 = 7;
    }
    else
    {
      v43 = *(_DWORD *)(a1 + 368);
      v31 = *(_QWORD *)(a1 + 360);
      if ( !v43 )
        goto LABEL_70;
      v32 = v43 - 1;
    }
    v33 = v11[1];
    v34 = v32 & (37 * v33);
    v26 = v31 + 24LL * v34;
    v35 = *(_DWORD *)v26;
    if ( *(_DWORD *)v26 != v33 )
    {
      v46 = 1;
      v42 = 0;
      while ( v35 != -1 )
      {
        if ( !v42 && v35 == -2 )
          v42 = v26;
        v34 = v32 & (v46 + v34);
        v26 = v31 + 24LL * v34;
        v35 = *(_DWORD *)v26;
        if ( v33 == *(_DWORD *)v26 )
          goto LABEL_40;
        ++v46;
      }
      goto LABEL_46;
    }
LABEL_40:
    v25 = *(_DWORD *)(a1 + 352);
    goto LABEL_28;
  }
  if ( v15 - *(_DWORD *)(a1 + 356) - v27 <= v15 >> 3 )
  {
    sub_200F500(a1 + 344, v15);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v36 = a1 + 360;
      v37 = 7;
      goto LABEL_43;
    }
    v44 = *(_DWORD *)(a1 + 368);
    v36 = *(_QWORD *)(a1 + 360);
    if ( v44 )
    {
      v37 = v44 - 1;
LABEL_43:
      v38 = v11[1];
      v39 = v37 & (37 * v38);
      v26 = v36 + 24LL * v39;
      v40 = *(_DWORD *)v26;
      if ( *(_DWORD *)v26 != v38 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -1 )
        {
          if ( !v42 && v40 == -2 )
            v42 = v26;
          v39 = v37 & (v41 + v39);
          v26 = v36 + 24LL * v39;
          v40 = *(_DWORD *)v26;
          if ( v38 == *(_DWORD *)v26 )
            goto LABEL_40;
          ++v41;
        }
LABEL_46:
        if ( v42 )
          v26 = v42;
        goto LABEL_40;
      }
      goto LABEL_40;
    }
LABEL_70:
    *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
    BUG();
  }
LABEL_28:
  *(_DWORD *)(a1 + 352) = (2 * (v25 >> 1) + 2) | v25 & 1;
  if ( *(_DWORD *)v26 != -1 )
    --*(_DWORD *)(a1 + 356);
  v29 = v11[1];
  *(_QWORD *)(v26 + 8) = 0;
  *(_DWORD *)(v26 + 16) = 0;
  *(_DWORD *)v26 = v29;
  return 0;
}
