// Function: sub_2138AD0
// Address: 0x2138ad0
//
__int64 __fastcall sub_2138AD0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v4; // r12d
  char v5; // di
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // eax
  _DWORD *v9; // r13
  int v10; // edx
  char v11; // di
  __int64 v12; // r10
  int v13; // esi
  int v14; // eax
  unsigned int v15; // edx
  __int64 v16; // rcx
  int v17; // r9d
  unsigned int v19; // esi
  unsigned int v20; // esi
  unsigned int v21; // eax
  __int64 v22; // r8
  int v23; // edx
  unsigned int v24; // r9d
  __int64 v25; // rdi
  int v26; // eax
  unsigned int v27; // eax
  int v28; // edx
  unsigned int v29; // r8d
  __int64 v30; // rdi
  int v31; // r11d
  int v32; // r9d
  _DWORD *v33; // r8
  __int64 v34; // rdi
  int v35; // edx
  int v36; // eax
  unsigned int v37; // esi
  int v38; // ecx
  __int64 v39; // rcx
  int v40; // eax
  unsigned int v41; // edx
  int v42; // esi
  __int64 v43; // rcx
  int v44; // edx
  unsigned int v45; // eax
  int v46; // esi
  int v47; // r8d
  _DWORD *v48; // rdi
  __int64 v49; // rdi
  int v50; // ecx
  int v51; // edx
  unsigned int v52; // esi
  int v53; // eax
  int v54; // r10d
  __int64 v55; // r9
  int v56; // eax
  int v57; // edx
  int v58; // ecx
  int v59; // edx
  __int64 v60; // r12
  int v61; // r8d
  int v62; // r10d

  v4 = sub_200F8F0(a1, a2, a3);
  v5 = *(_BYTE *)(a1 + 560) & 1;
  if ( (*(_BYTE *)(a1 + 560) & 1) != 0 )
  {
    v6 = a1 + 568;
    v7 = 7;
  }
  else
  {
    v19 = *(_DWORD *)(a1 + 576);
    v6 = *(_QWORD *)(a1 + 568);
    if ( !v19 )
    {
      v27 = *(_DWORD *)(a1 + 560);
      ++*(_QWORD *)(a1 + 552);
      v9 = 0;
      v28 = (v27 >> 1) + 1;
LABEL_20:
      v29 = 3 * v19;
      goto LABEL_21;
    }
    v7 = v19 - 1;
  }
  v8 = v7 & (37 * v4);
  v9 = (_DWORD *)(v6 + 8LL * v8);
  v10 = *v9;
  if ( v4 == *v9 )
    goto LABEL_4;
  v32 = 1;
  v33 = 0;
  while ( v10 != -1 )
  {
    if ( !v33 && v10 == -2 )
      v33 = v9;
    v8 = v7 & (v32 + v8);
    v9 = (_DWORD *)(v6 + 8LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
      goto LABEL_4;
    ++v32;
  }
  v27 = *(_DWORD *)(a1 + 560);
  v19 = 8;
  if ( v33 )
    v9 = v33;
  ++*(_QWORD *)(a1 + 552);
  v29 = 24;
  v28 = (v27 >> 1) + 1;
  if ( !v5 )
  {
    v19 = *(_DWORD *)(a1 + 576);
    goto LABEL_20;
  }
LABEL_21:
  v30 = a1 + 552;
  if ( 4 * v28 >= v29 )
  {
    sub_20108A0(v30, 2 * v19);
    if ( (*(_BYTE *)(a1 + 560) & 1) != 0 )
    {
      v39 = a1 + 568;
      v40 = 7;
    }
    else
    {
      v56 = *(_DWORD *)(a1 + 576);
      v39 = *(_QWORD *)(a1 + 568);
      if ( !v56 )
        goto LABEL_104;
      v40 = v56 - 1;
    }
    v41 = v40 & (37 * v4);
    v9 = (_DWORD *)(v39 + 8LL * v41);
    v42 = *v9;
    if ( v4 != *v9 )
    {
      v61 = 1;
      v48 = 0;
      while ( v42 != -1 )
      {
        if ( !v48 && v42 == -2 )
          v48 = v9;
        v41 = v40 & (v61 + v41);
        v9 = (_DWORD *)(v39 + 8LL * v41);
        v42 = *v9;
        if ( v4 == *v9 )
          goto LABEL_45;
        ++v61;
      }
      goto LABEL_51;
    }
LABEL_45:
    v27 = *(_DWORD *)(a1 + 560);
    goto LABEL_23;
  }
  if ( v19 - *(_DWORD *)(a1 + 564) - v28 <= v19 >> 3 )
  {
    sub_20108A0(v30, v19);
    if ( (*(_BYTE *)(a1 + 560) & 1) != 0 )
    {
      v43 = a1 + 568;
      v44 = 7;
      goto LABEL_48;
    }
    v59 = *(_DWORD *)(a1 + 576);
    v43 = *(_QWORD *)(a1 + 568);
    if ( v59 )
    {
      v44 = v59 - 1;
LABEL_48:
      v45 = v44 & (37 * v4);
      v9 = (_DWORD *)(v43 + 8LL * v45);
      v46 = *v9;
      if ( v4 != *v9 )
      {
        v47 = 1;
        v48 = 0;
        while ( v46 != -1 )
        {
          if ( !v48 && v46 == -2 )
            v48 = v9;
          v45 = v44 & (v47 + v45);
          v9 = (_DWORD *)(v43 + 8LL * v45);
          v46 = *v9;
          if ( v4 == *v9 )
            goto LABEL_45;
          ++v47;
        }
LABEL_51:
        if ( v48 )
          v9 = v48;
        goto LABEL_45;
      }
      goto LABEL_45;
    }
LABEL_104:
    *(_DWORD *)(a1 + 560) = (2 * (*(_DWORD *)(a1 + 560) >> 1) + 2) | *(_DWORD *)(a1 + 560) & 1;
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 560) = (2 * (v27 >> 1) + 2) | v27 & 1;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 564);
  *v9 = v4;
  v9[1] = 0;
LABEL_4:
  sub_200D1B0(a1, v9 + 1);
  v11 = *(_BYTE *)(a1 + 352) & 1;
  if ( v11 )
  {
    v12 = a1 + 360;
    v13 = 7;
  }
  else
  {
    v20 = *(_DWORD *)(a1 + 368);
    v12 = *(_QWORD *)(a1 + 360);
    if ( !v20 )
    {
      v21 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v22 = 0;
      v23 = (v21 >> 1) + 1;
LABEL_13:
      v24 = 3 * v20;
      goto LABEL_14;
    }
    v13 = v20 - 1;
  }
  v14 = v9[1];
  v15 = v13 & (37 * v14);
  v16 = v12 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *(_DWORD *)v16 == v14 )
    return *(_QWORD *)(v16 + 8);
  v31 = 1;
  v22 = 0;
  while ( v17 != -1 )
  {
    if ( v22 || v17 != -2 )
      v16 = v22;
    v15 = v13 & (v31 + v15);
    v60 = v12 + 24LL * v15;
    v17 = *(_DWORD *)v60;
    if ( v14 == *(_DWORD *)v60 )
      return *(_QWORD *)(v60 + 8);
    v22 = v16;
    ++v31;
    v16 = v12 + 24LL * v15;
  }
  v21 = *(_DWORD *)(a1 + 352);
  v24 = 24;
  v20 = 8;
  if ( !v22 )
    v22 = v16;
  ++*(_QWORD *)(a1 + 344);
  v23 = (v21 >> 1) + 1;
  if ( !v11 )
  {
    v20 = *(_DWORD *)(a1 + 368);
    goto LABEL_13;
  }
LABEL_14:
  v25 = a1 + 344;
  if ( v24 <= 4 * v23 )
  {
    sub_200F500(v25, 2 * v20);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v34 = a1 + 360;
      v35 = 7;
    }
    else
    {
      v57 = *(_DWORD *)(a1 + 368);
      v34 = *(_QWORD *)(a1 + 360);
      if ( !v57 )
        goto LABEL_103;
      v35 = v57 - 1;
    }
    v36 = v9[1];
    v37 = v35 & (37 * v36);
    v22 = v34 + 24LL * v37;
    v38 = *(_DWORD *)v22;
    if ( *(_DWORD *)v22 != v36 )
    {
      v62 = 1;
      v55 = 0;
      while ( v38 != -1 )
      {
        if ( !v55 && v38 == -2 )
          v55 = v22;
        v37 = v35 & (v62 + v37);
        v22 = v34 + 24LL * v37;
        v38 = *(_DWORD *)v22;
        if ( v36 == *(_DWORD *)v22 )
          goto LABEL_41;
        ++v62;
      }
      goto LABEL_58;
    }
LABEL_41:
    v21 = *(_DWORD *)(a1 + 352);
    goto LABEL_16;
  }
  if ( v20 - *(_DWORD *)(a1 + 356) - v23 <= v20 >> 3 )
  {
    sub_200F500(v25, v20);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v49 = a1 + 360;
      v50 = 7;
      goto LABEL_55;
    }
    v58 = *(_DWORD *)(a1 + 368);
    v49 = *(_QWORD *)(a1 + 360);
    if ( v58 )
    {
      v50 = v58 - 1;
LABEL_55:
      v51 = v9[1];
      v52 = v50 & (37 * v51);
      v22 = v49 + 24LL * v52;
      v53 = *(_DWORD *)v22;
      if ( *(_DWORD *)v22 != v51 )
      {
        v54 = 1;
        v55 = 0;
        while ( v53 != -1 )
        {
          if ( v53 == -2 && !v55 )
            v55 = v22;
          v52 = v50 & (v54 + v52);
          v22 = v49 + 24LL * v52;
          v53 = *(_DWORD *)v22;
          if ( v51 == *(_DWORD *)v22 )
            goto LABEL_41;
          ++v54;
        }
LABEL_58:
        if ( v55 )
          v22 = v55;
        goto LABEL_41;
      }
      goto LABEL_41;
    }
LABEL_103:
    *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
    BUG();
  }
LABEL_16:
  *(_DWORD *)(a1 + 352) = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( *(_DWORD *)v22 != -1 )
    --*(_DWORD *)(a1 + 356);
  v26 = v9[1];
  *(_QWORD *)(v22 + 8) = 0;
  *(_DWORD *)(v22 + 16) = 0;
  *(_DWORD *)v22 = v26;
  return 0;
}
