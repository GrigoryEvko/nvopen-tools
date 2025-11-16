// Function: sub_375F970
// Address: 0x375f970
//
__int64 __fastcall sub_375F970(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  int v7; // ebx
  char v8; // r8
  __int64 v9; // rdi
  int v10; // esi
  unsigned int v11; // edx
  _DWORD *v12; // rax
  int v13; // r9d
  _DWORD *v14; // rbx
  __int64 result; // rax
  unsigned int v16; // esi
  unsigned int v17; // edx
  _DWORD *v18; // rcx
  int v19; // eax
  unsigned int v20; // r9d
  int v21; // r10d
  __int64 v22; // rsi
  int v23; // eax
  unsigned int v24; // edx
  int v25; // edi
  __int64 v26; // rsi
  int v27; // edx
  unsigned int v28; // eax
  int v29; // edi
  int v30; // r9d
  _DWORD *v31; // r8
  int v32; // eax
  int v33; // edx
  int v34; // r9d
  unsigned __int64 v35; // [rsp+0h] [rbp-30h] BYREF
  __int64 v36; // [rsp+8h] [rbp-28h]

  v36 = a5;
  v35 = a4;
  sub_375EAB0(a1, (__int64)&v35);
  v7 = sub_375D5B0(a1, a2, a3);
  v8 = *(_BYTE *)(a1 + 1072) & 1;
  if ( v8 )
  {
    v9 = a1 + 1080;
    v10 = 7;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 1088);
    v9 = *(_QWORD *)(a1 + 1080);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 1072);
      v18 = 0;
      ++*(_QWORD *)(a1 + 1064);
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v10 = v16 - 1;
  }
  v11 = v10 & (37 * v7);
  v12 = (_DWORD *)(v9 + 8LL * v11);
  v13 = *v12;
  if ( v7 == *v12 )
  {
LABEL_4:
    v14 = v12 + 1;
    goto LABEL_5;
  }
  v21 = 1;
  v18 = 0;
  while ( v13 != -1 )
  {
    if ( !v18 && v13 == -2 )
      v18 = v12;
    v11 = v10 & (v21 + v11);
    v12 = (_DWORD *)(v9 + 8LL * v11);
    v13 = *v12;
    if ( v7 == *v12 )
      goto LABEL_4;
    ++v21;
  }
  v17 = *(_DWORD *)(a1 + 1072);
  v20 = 24;
  v16 = 8;
  if ( !v18 )
    v18 = v12;
  ++*(_QWORD *)(a1 + 1064);
  v19 = (v17 >> 1) + 1;
  if ( !v8 )
  {
    v16 = *(_DWORD *)(a1 + 1088);
    goto LABEL_9;
  }
LABEL_10:
  if ( 4 * v19 >= v20 )
  {
    sub_375BDE0(a1 + 1064, 2 * v16);
    if ( (*(_BYTE *)(a1 + 1072) & 1) != 0 )
    {
      v22 = a1 + 1080;
      v23 = 7;
    }
    else
    {
      v32 = *(_DWORD *)(a1 + 1088);
      v22 = *(_QWORD *)(a1 + 1080);
      if ( !v32 )
        goto LABEL_53;
      v23 = v32 - 1;
    }
    v24 = v23 & (37 * v7);
    v18 = (_DWORD *)(v22 + 8LL * v24);
    v25 = *v18;
    if ( v7 != *v18 )
    {
      v34 = 1;
      v31 = 0;
      while ( v25 != -1 )
      {
        if ( !v31 && v25 == -2 )
          v31 = v18;
        v24 = v23 & (v34 + v24);
        v18 = (_DWORD *)(v22 + 8LL * v24);
        v25 = *v18;
        if ( v7 == *v18 )
          goto LABEL_24;
        ++v34;
      }
      goto LABEL_30;
    }
LABEL_24:
    v17 = *(_DWORD *)(a1 + 1072);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 1076) - v19 <= v16 >> 3 )
  {
    sub_375BDE0(a1 + 1064, v16);
    if ( (*(_BYTE *)(a1 + 1072) & 1) != 0 )
    {
      v26 = a1 + 1080;
      v27 = 7;
      goto LABEL_27;
    }
    v33 = *(_DWORD *)(a1 + 1088);
    v26 = *(_QWORD *)(a1 + 1080);
    if ( v33 )
    {
      v27 = v33 - 1;
LABEL_27:
      v28 = v27 & (37 * v7);
      v18 = (_DWORD *)(v26 + 8LL * v28);
      v29 = *v18;
      if ( v7 != *v18 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -1 )
        {
          if ( v29 == -2 && !v31 )
            v31 = v18;
          v28 = v27 & (v30 + v28);
          v18 = (_DWORD *)(v26 + 8LL * v28);
          v29 = *v18;
          if ( v7 == *v18 )
            goto LABEL_24;
          ++v30;
        }
LABEL_30:
        if ( v31 )
          v18 = v31;
        goto LABEL_24;
      }
      goto LABEL_24;
    }
LABEL_53:
    *(_DWORD *)(a1 + 1072) = (2 * (*(_DWORD *)(a1 + 1072) >> 1) + 2) | *(_DWORD *)(a1 + 1072) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 1072) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *v18 != -1 )
    --*(_DWORD *)(a1 + 1076);
  *v18 = v7;
  v14 = v18 + 1;
  v18[1] = 0;
LABEL_5:
  result = sub_375D5B0(a1, v35, v36);
  *v14 = result;
  return result;
}
