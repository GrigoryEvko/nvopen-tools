// Function: sub_12A5B50
// Address: 0x12a5b50
//
__int64 __fastcall sub_12A5B50(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // r11d
  _QWORD *v12; // r13
  int v13; // eax
  int v14; // edx
  const char *v15; // rsi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r9d
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // r8d
  unsigned int v27; // r14d
  _QWORD *v28; // rdi
  __int64 v29; // rcx

  v3 = a1 + 376;
  v5 = *(_DWORD *)(a1 + 400);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 376);
    goto LABEL_18;
  }
  v6 = *(_QWORD *)(a1 + 384);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -8 )
    {
      if ( !v12 && v9 == -16 )
        v12 = v8;
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (_QWORD *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    v13 = *(_DWORD *)(a1 + 392);
    ++*(_QWORD *)(a1 + 376);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 396) - v14 > v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 392) = v14;
        if ( *v12 != -8 )
          --*(_DWORD *)(a1 + 396);
        *v12 = a2;
        v12[1] = 0;
        goto LABEL_14;
      }
      sub_12A5990(v3, v5);
      v23 = *(_DWORD *)(a1 + 400);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 384);
        v26 = 1;
        v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v14 = *(_DWORD *)(a1 + 392) + 1;
        v28 = 0;
        v12 = (_QWORD *)(v25 + 16LL * v27);
        v29 = *v12;
        if ( *v12 != a2 )
        {
          while ( v29 != -8 )
          {
            if ( !v28 && v29 == -16 )
              v28 = v12;
            v27 = v24 & (v26 + v27);
            v12 = (_QWORD *)(v25 + 16LL * v27);
            v29 = *v12;
            if ( *v12 == a2 )
              goto LABEL_11;
            ++v26;
          }
          if ( v28 )
            v12 = v28;
        }
        goto LABEL_11;
      }
LABEL_47:
      ++*(_DWORD *)(a1 + 392);
      BUG();
    }
LABEL_18:
    sub_12A5990(v3, 2 * v5);
    v16 = *(_DWORD *)(a1 + 400);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 384);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 392) + 1;
      v12 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v12;
      if ( *v12 != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = v12;
          v19 = v17 & (v21 + v19);
          v12 = (_QWORD *)(v18 + 16LL * v19);
          v20 = *v12;
          if ( *v12 == a2 )
            goto LABEL_11;
          ++v21;
        }
        if ( v22 )
          v12 = v22;
      }
      goto LABEL_11;
    }
    goto LABEL_47;
  }
LABEL_3:
  if ( v8[1] )
    return v8[1];
  v12 = v8;
LABEL_14:
  v15 = *(const char **)(*(_QWORD *)(a2 + 72) + 8LL);
  if ( !v15 )
    v15 = "compiler_generated_label";
  result = sub_12A4D50(a1, (__int64)v15, 0, 0);
  v12[1] = result;
  return result;
}
