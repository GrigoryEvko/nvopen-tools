// Function: sub_3760810
// Address: 0x3760810
//
__int64 __fastcall sub_3760810(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8)
{
  int v10; // ebx
  char v11; // dl
  __int64 v12; // r9
  int v13; // esi
  unsigned int v14; // ecx
  _DWORD *v15; // rax
  int v16; // edi
  _DWORD *v17; // rbx
  __int64 result; // rax
  unsigned int v19; // esi
  unsigned int v20; // eax
  _DWORD *v21; // r8
  int v22; // ecx
  unsigned int v23; // edi
  int v24; // r10d
  __int64 v25; // rsi
  int v26; // eax
  unsigned int v27; // edx
  int v28; // ecx
  __int64 v29; // rsi
  int v30; // ecx
  unsigned int v31; // eax
  int v32; // edx
  int v33; // r9d
  _DWORD *v34; // rdi
  int v35; // eax
  int v36; // ecx
  int v37; // r9d
  unsigned __int64 v38; // [rsp+0h] [rbp-30h] BYREF
  __int64 v39; // [rsp+8h] [rbp-28h]

  v38 = a4;
  v39 = a5;
  sub_375EAB0(a1, (__int64)&v38);
  sub_375EAB0(a1, (__int64)&a7);
  v10 = sub_375D5B0(a1, a2, a3);
  v11 = *(_BYTE *)(a1 + 1344) & 1;
  if ( v11 )
  {
    v12 = a1 + 1352;
    v13 = 7;
  }
  else
  {
    v19 = *(_DWORD *)(a1 + 1360);
    v12 = *(_QWORD *)(a1 + 1352);
    if ( !v19 )
    {
      v20 = *(_DWORD *)(a1 + 1344);
      v21 = 0;
      ++*(_QWORD *)(a1 + 1336);
      v22 = (v20 >> 1) + 1;
LABEL_9:
      v23 = 3 * v19;
      goto LABEL_10;
    }
    v13 = v19 - 1;
  }
  v14 = v13 & (37 * v10);
  v15 = (_DWORD *)(v12 + 12LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_4:
    v17 = v15 + 1;
    goto LABEL_5;
  }
  v24 = 1;
  v21 = 0;
  while ( v16 != -1 )
  {
    if ( !v21 && v16 == -2 )
      v21 = v15;
    v14 = v13 & (v24 + v14);
    v15 = (_DWORD *)(v12 + 12LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_4;
    ++v24;
  }
  v23 = 24;
  v19 = 8;
  if ( !v21 )
    v21 = v15;
  v20 = *(_DWORD *)(a1 + 1344);
  ++*(_QWORD *)(a1 + 1336);
  v22 = (v20 >> 1) + 1;
  if ( !v11 )
  {
    v19 = *(_DWORD *)(a1 + 1360);
    goto LABEL_9;
  }
LABEL_10:
  if ( 4 * v22 >= v23 )
  {
    sub_375C1D0(a1 + 1336, 2 * v19);
    if ( (*(_BYTE *)(a1 + 1344) & 1) != 0 )
    {
      v25 = a1 + 1352;
      v26 = 7;
    }
    else
    {
      v35 = *(_DWORD *)(a1 + 1360);
      v25 = *(_QWORD *)(a1 + 1352);
      if ( !v35 )
        goto LABEL_53;
      v26 = v35 - 1;
    }
    v27 = v26 & (37 * v10);
    v21 = (_DWORD *)(v25 + 12LL * v27);
    v28 = *v21;
    if ( v10 != *v21 )
    {
      v37 = 1;
      v34 = 0;
      while ( v28 != -1 )
      {
        if ( !v34 && v28 == -2 )
          v34 = v21;
        v27 = v26 & (v37 + v27);
        v21 = (_DWORD *)(v25 + 12LL * v27);
        v28 = *v21;
        if ( v10 == *v21 )
          goto LABEL_24;
        ++v37;
      }
      goto LABEL_30;
    }
LABEL_24:
    v20 = *(_DWORD *)(a1 + 1344);
    goto LABEL_12;
  }
  if ( v19 - *(_DWORD *)(a1 + 1348) - v22 <= v19 >> 3 )
  {
    sub_375C1D0(a1 + 1336, v19);
    if ( (*(_BYTE *)(a1 + 1344) & 1) != 0 )
    {
      v29 = a1 + 1352;
      v30 = 7;
      goto LABEL_27;
    }
    v36 = *(_DWORD *)(a1 + 1360);
    v29 = *(_QWORD *)(a1 + 1352);
    if ( v36 )
    {
      v30 = v36 - 1;
LABEL_27:
      v31 = v30 & (37 * v10);
      v21 = (_DWORD *)(v29 + 12LL * v31);
      v32 = *v21;
      if ( v10 != *v21 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -1 )
        {
          if ( v32 == -2 && !v34 )
            v34 = v21;
          v31 = v30 & (v33 + v31);
          v21 = (_DWORD *)(v29 + 12LL * v31);
          v32 = *v21;
          if ( v10 == *v21 )
            goto LABEL_24;
          ++v33;
        }
LABEL_30:
        if ( v34 )
          v21 = v34;
        goto LABEL_24;
      }
      goto LABEL_24;
    }
LABEL_53:
    *(_DWORD *)(a1 + 1344) = (2 * (*(_DWORD *)(a1 + 1344) >> 1) + 2) | *(_DWORD *)(a1 + 1344) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 1344) = (2 * (v20 >> 1) + 2) | v20 & 1;
  if ( *v21 != -1 )
    --*(_DWORD *)(a1 + 1348);
  *v21 = v10;
  v17 = v21 + 1;
  *(_QWORD *)(v21 + 1) = 0;
LABEL_5:
  *v17 = sub_375D5B0(a1, v38, v39);
  result = sub_375D5B0(a1, a7, a8);
  v17[1] = result;
  return result;
}
