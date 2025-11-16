// Function: sub_2010110
// Address: 0x2010110
//
__int64 __fastcall sub_2010110(__int64 a1, __int64 a2)
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
  int v15; // ecx
  int v16; // r11d
  __int64 v17; // r10
  __int64 v18; // r8
  int v19; // esi
  int v20; // edx
  unsigned int v21; // edi
  __int64 v22; // r8
  int v23; // esi
  unsigned int v24; // edi
  int v25; // r10d
  __int64 v26; // r9
  int v27; // esi
  int v28; // esi
  int v29; // r10d
  int v30[9]; // [rsp+Ch] [rbp-24h] BYREF

  v30[0] = sub_200F8F0(a1, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
  sub_200D1B0(a1, v30);
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
  v7 = v6 & (37 * v30[0]);
  v8 = v5 + 24LL * v7;
  v9 = *(_DWORD *)v8;
  if ( v30[0] == *(_DWORD *)v8 )
    goto LABEL_4;
  v16 = 1;
  v17 = 0;
  while ( v9 != -1 )
  {
    if ( !v17 && v9 == -2 )
      v17 = v8;
    v7 = v6 & (v16 + v7);
    v8 = v5 + 24LL * v7;
    v9 = *(_DWORD *)v8;
    if ( v30[0] == *(_DWORD *)v8 )
      goto LABEL_4;
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
      v27 = *(_DWORD *)(a1 + 368);
      v18 = *(_QWORD *)(a1 + 360);
      if ( !v27 )
        goto LABEL_53;
      v19 = v27 - 1;
    }
    v20 = v30[0];
    v21 = v19 & (37 * v30[0]);
    v8 = v18 + 24LL * v21;
    v15 = *(_DWORD *)v8;
    if ( v30[0] != *(_DWORD *)v8 )
    {
      v29 = 1;
      v26 = 0;
      while ( v15 != -1 )
      {
        if ( !v26 && v15 == -2 )
          v26 = v8;
        v21 = v19 & (v29 + v21);
        v8 = v18 + 24LL * v21;
        v15 = *(_DWORD *)v8;
        if ( v30[0] == *(_DWORD *)v8 )
          goto LABEL_24;
        ++v29;
      }
      goto LABEL_30;
    }
LABEL_24:
    v12 = *(_DWORD *)(a1 + 352);
    goto LABEL_12;
  }
  if ( v11 - *(_DWORD *)(a1 + 356) - v13 <= v11 >> 3 )
  {
    sub_200F500(a1 + 344, v11);
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v22 = a1 + 360;
      v23 = 7;
      goto LABEL_27;
    }
    v28 = *(_DWORD *)(a1 + 368);
    v22 = *(_QWORD *)(a1 + 360);
    if ( v28 )
    {
      v23 = v28 - 1;
LABEL_27:
      v20 = v30[0];
      v24 = v23 & (37 * v30[0]);
      v8 = v22 + 24LL * v24;
      v15 = *(_DWORD *)v8;
      if ( v30[0] != *(_DWORD *)v8 )
      {
        v25 = 1;
        v26 = 0;
        while ( v15 != -1 )
        {
          if ( v15 == -2 && !v26 )
            v26 = v8;
          v24 = v23 & (v25 + v24);
          v8 = v22 + 24LL * v24;
          v15 = *(_DWORD *)v8;
          if ( v30[0] == *(_DWORD *)v8 )
            goto LABEL_24;
          ++v25;
        }
LABEL_30:
        v15 = v20;
        if ( v26 )
          v8 = v26;
        goto LABEL_24;
      }
      goto LABEL_24;
    }
LABEL_53:
    JUMPOUT(0x42362E);
  }
  v15 = v30[0];
LABEL_12:
  *(_DWORD *)(a1 + 352) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *(_DWORD *)v8 != -1 )
    --*(_DWORD *)(a1 + 356);
  *(_DWORD *)v8 = v15;
  *(_QWORD *)(v8 + 8) = 0;
  *(_DWORD *)(v8 + 16) = 0;
LABEL_4:
  *(_QWORD *)a2 = *(_QWORD *)(v8 + 8);
  result = *(unsigned int *)(v8 + 16);
  *(_DWORD *)(a2 + 8) = result;
  return result;
}
