// Function: sub_25CF280
// Address: 0x25cf280
//
__int64 __fastcall sub_25CF280(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rcx
  _QWORD *v9; // r11
  int v10; // r14d
  unsigned __int64 v11; // rax
  unsigned int v12; // r9d
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  char v16; // dl
  int v18; // eax
  int v19; // edx
  int v20; // eax
  int v21; // ecx
  __int64 v22; // r9
  unsigned __int64 v23; // rsi
  unsigned int v24; // r8d
  unsigned __int64 v25; // rax
  int v26; // r11d
  _QWORD *v27; // r10
  int v28; // eax
  int v29; // ecx
  __int64 v30; // r9
  int v31; // r11d
  unsigned __int64 v32; // rsi
  unsigned int v33; // r8d
  unsigned __int64 v34; // rax

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_19;
  }
  v8 = *(_QWORD *)(a2 + 8);
  v9 = 0;
  v10 = 1;
  v11 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = v11 & (v6 - 1);
  v13 = (_QWORD *)(v8 + 8LL * v12);
  v14 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == v14 )
  {
LABEL_3:
    v15 = v8 + 8 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -8 )
  {
    if ( !v9 && v14 == -16 )
      v9 = v13;
    v12 = (v6 - 1) & (v10 + v12);
    v13 = (_QWORD *)(v8 + 8LL * v12);
    v14 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == v14 )
      goto LABEL_3;
    ++v10;
  }
  *(_QWORD *)a2 = v7 + 1;
  v18 = *(_DWORD *)(a2 + 16);
  if ( v9 )
    v13 = v9;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v6) )
  {
LABEL_19:
    sub_BB0720(a2, 2 * v6);
    v20 = *(_DWORD *)(a2 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a2 + 8);
      v23 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
      v24 = v23 & (v20 - 1);
      v19 = *(_DWORD *)(a2 + 16) + 1;
      v13 = (_QWORD *)(v22 + 8LL * v24);
      v25 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v23 == v25 )
        goto LABEL_15;
      v26 = 1;
      v27 = 0;
      while ( v25 != -8 )
      {
        if ( !v27 && v25 == -16 )
          v27 = v13;
        v24 = v21 & (v26 + v24);
        v13 = (_QWORD *)(v22 + 8LL * v24);
        v25 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v23 == v25 )
          goto LABEL_15;
        ++v26;
      }
LABEL_23:
      if ( v27 )
        v13 = v27;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v6 >> 3 )
  {
    sub_BB0720(a2, v6);
    v28 = *(_DWORD *)(a2 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a2 + 8);
      v27 = 0;
      v31 = 1;
      v32 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
      v33 = v32 & (v28 - 1);
      v19 = *(_DWORD *)(a2 + 16) + 1;
      v13 = (_QWORD *)(v30 + 8LL * v33);
      v34 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v32 == v34 )
        goto LABEL_15;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v27 )
          v27 = v13;
        v33 = v29 & (v31 + v33);
        v13 = (_QWORD *)(v30 + 8LL * v33);
        v34 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v32 == v34 )
          goto LABEL_15;
        ++v31;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v19;
  if ( (*v13 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  v16 = 1;
LABEL_4:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v15;
  *(_BYTE *)(a1 + 32) = v16;
  return a1;
}
