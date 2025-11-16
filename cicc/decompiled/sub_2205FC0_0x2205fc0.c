// Function: sub_2205FC0
// Address: 0x2205fc0
//
__int64 __fastcall sub_2205FC0(__int64 a1, __int64 a2, int *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  int v8; // eax
  __int64 v9; // r9
  int v10; // r14d
  int *v11; // r11
  unsigned int v12; // ecx
  int *v13; // rdx
  int v14; // edi
  __int64 v15; // rsi
  char v16; // cl
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // eax
  int v24; // r8d
  int v25; // r11d
  int *v26; // r10
  int v27; // eax
  int v28; // eax
  __int64 v29; // r9
  int v30; // r11d
  unsigned int v31; // r8d
  int v32; // edi

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_19;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v6 - 1) & (37 * *a3);
  v13 = (int *)(v9 + 4LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    v15 = v9 + 4 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -1 )
  {
    if ( !v11 && v14 == -2 )
      v11 = v13;
    v12 = (v6 - 1) & (v10 + v12);
    v13 = (int *)(v9 + 4LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v10;
  }
  *(_QWORD *)a2 = v7 + 1;
  v18 = *(_DWORD *)(a2 + 16);
  if ( v11 )
    v13 = v11;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_19:
    sub_136B240(a2, 2 * v6);
    v20 = *(_DWORD *)(a2 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a2 + 8);
      v19 = *(_DWORD *)(a2 + 16) + 1;
      v23 = (v20 - 1) & (37 * *a3);
      v13 = (int *)(v22 + 4LL * (v21 & (unsigned int)(37 * *a3)));
      v24 = *v13;
      if ( *v13 == *a3 )
        goto LABEL_15;
      v25 = 1;
      v26 = 0;
      while ( v24 != -1 )
      {
        if ( !v26 && v24 == -2 )
          v26 = v13;
        v23 = v21 & (v25 + v23);
        v13 = (int *)(v22 + 4LL * v23);
        v24 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v25;
      }
LABEL_23:
      if ( v26 )
        v13 = v26;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v6 >> 3 )
  {
    sub_136B240(a2, v6);
    v27 = *(_DWORD *)(a2 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a2 + 8);
      v26 = 0;
      v30 = 1;
      v19 = *(_DWORD *)(a2 + 16) + 1;
      v31 = v28 & (37 * *a3);
      v13 = (int *)(v29 + 4LL * v31);
      v32 = *v13;
      if ( *a3 == *v13 )
        goto LABEL_15;
      while ( v32 != -1 )
      {
        if ( v32 == -2 && !v26 )
          v26 = v13;
        v31 = v28 & (v30 + v31);
        v13 = (int *)(v29 + 4LL * v31);
        v32 = *v13;
        if ( *a3 == *v13 )
          goto LABEL_15;
        ++v30;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 4LL * *(unsigned int *)(a2 + 24);
  v16 = 1;
LABEL_4:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v15;
  *(_BYTE *)(a1 + 32) = v16;
  return a1;
}
