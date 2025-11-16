// Function: sub_21387B0
// Address: 0x21387b0
//
__int64 __fastcall sub_21387B0(__int64 a1, int *a2)
{
  char v4; // dl
  __int64 v5; // r9
  int v6; // esi
  unsigned int v7; // edi
  __int64 v8; // rax
  int v9; // r8d
  __int64 result; // rax
  unsigned int v11; // esi
  unsigned int v12; // r8d
  int v13; // ecx
  unsigned int v14; // edi
  int v15; // edx
  int v16; // r11d
  __int64 v17; // r10
  __int64 v18; // r8
  int v19; // ecx
  unsigned int v20; // esi
  int v21; // edi
  __int64 v22; // r8
  int v23; // ecx
  unsigned int v24; // esi
  int v25; // edi
  int v26; // r10d
  __int64 v27; // r9
  int v28; // ecx
  int v29; // ecx
  int v30; // r10d

  sub_200D1B0(a1, a2);
  v4 = *(_BYTE *)(a1 + 352) & 1;
  if ( v4 )
  {
    v5 = a1 + 360;
    v6 = 7;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 368);
    v5 = *(_QWORD *)(a1 + 360);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v8 = 0;
      v13 = (v12 >> 1) + 1;
LABEL_8:
      v14 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (37 * *a2);
  v8 = v5 + 24LL * v7;
  v9 = *(_DWORD *)v8;
  if ( *a2 == *(_DWORD *)v8 )
    return v8 + 8;
  v16 = 1;
  v17 = 0;
  while ( v9 != -1 )
  {
    if ( !v17 && v9 == -2 )
      v17 = v8;
    v7 = v6 & (v16 + v7);
    v8 = v5 + 24LL * v7;
    v9 = *(_DWORD *)v8;
    if ( *a2 == *(_DWORD *)v8 )
      return v8 + 8;
    ++v16;
  }
  v12 = *(_DWORD *)(a1 + 352);
  v14 = 24;
  v11 = 8;
  if ( v17 )
    v8 = v17;
  ++*(_QWORD *)(a1 + 344);
  v13 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = *(_DWORD *)(a1 + 368);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v13 >= v14 )
  {
    sub_200F500(a1 + 344, 2 * v11);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v18 = a1 + 360;
      v19 = 7;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 368);
      v18 = *(_QWORD *)(a1 + 360);
      if ( !v28 )
        goto LABEL_52;
      v19 = v28 - 1;
    }
    v20 = v19 & (37 * *a2);
    v8 = v18 + 24LL * v20;
    v21 = *(_DWORD *)v8;
    if ( *a2 != *(_DWORD *)v8 )
    {
      v30 = 1;
      v27 = 0;
      while ( v21 != -1 )
      {
        if ( !v27 && v21 == -2 )
          v27 = v8;
        v20 = v19 & (v30 + v20);
        v8 = v18 + 24LL * v20;
        v21 = *(_DWORD *)v8;
        if ( *a2 == *(_DWORD *)v8 )
          goto LABEL_23;
        ++v30;
      }
      goto LABEL_29;
    }
LABEL_23:
    v12 = *(_DWORD *)(a1 + 352);
    goto LABEL_11;
  }
  if ( v11 - *(_DWORD *)(a1 + 356) - v13 <= v11 >> 3 )
  {
    sub_200F500(a1 + 344, v11);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v22 = a1 + 360;
      v23 = 7;
      goto LABEL_26;
    }
    v29 = *(_DWORD *)(a1 + 368);
    v22 = *(_QWORD *)(a1 + 360);
    if ( v29 )
    {
      v23 = v29 - 1;
LABEL_26:
      v24 = v23 & (37 * *a2);
      v8 = v22 + 24LL * v24;
      v25 = *(_DWORD *)v8;
      if ( *a2 != *(_DWORD *)v8 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -1 )
        {
          if ( v25 == -2 && !v27 )
            v27 = v8;
          v24 = v23 & (v26 + v24);
          v8 = v22 + 24LL * v24;
          v25 = *(_DWORD *)v8;
          if ( *a2 == *(_DWORD *)v8 )
            goto LABEL_23;
          ++v26;
        }
LABEL_29:
        if ( v27 )
          v8 = v27;
        goto LABEL_23;
      }
      goto LABEL_23;
    }
LABEL_52:
    *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 352) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *(_DWORD *)v8 != -1 )
    --*(_DWORD *)(a1 + 356);
  v15 = *a2;
  *(_QWORD *)(v8 + 8) = 0;
  result = v8 + 8;
  *(_DWORD *)(result + 8) = 0;
  *(_DWORD *)(result - 8) = v15;
  return result;
}
