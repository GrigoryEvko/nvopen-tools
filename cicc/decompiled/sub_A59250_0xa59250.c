// Function: sub_A59250
// Address: 0xa59250
//
unsigned __int64 __fastcall sub_A59250(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  unsigned int v4; // esi
  __int64 v6; // r8
  _QWORD *v7; // rdx
  int v8; // r11d
  unsigned int v9; // edi
  unsigned __int64 result; // rax
  __int64 v11; // rcx
  int v12; // eax
  int v13; // ecx
  int v14; // eax
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  _QWORD *v25; // r8
  unsigned int v26; // r13d
  int v27; // r9d
  __int64 v28; // rsi

  v2 = a1 + 224;
  v4 = *(_DWORD *)(a1 + 248);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 224);
    goto LABEL_17;
  }
  v6 = *(_QWORD *)(a1 + 232);
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v6 + 16LL * v9;
  v11 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
    return result;
  while ( v11 != -4 )
  {
    if ( v7 || v11 != -8 )
      result = (unsigned __int64)v7;
    v9 = (v4 - 1) & (v8 + v9);
    v11 = *(_QWORD *)(v6 + 16LL * v9);
    if ( v11 == a2 )
      return result;
    ++v8;
    v7 = (_QWORD *)result;
    result = v6 + 16LL * v9;
  }
  if ( !v7 )
    v7 = (_QWORD *)result;
  v12 = *(_DWORD *)(a1 + 240);
  ++*(_QWORD *)(a1 + 224);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_17:
    sub_A59070(v2, 2 * v4);
    v15 = *(_DWORD *)(a1 + 248);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 232);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 240) + 1;
      v7 = (_QWORD *)(v17 + 16LL * v18);
      v19 = *v7;
      if ( *v7 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4 )
        {
          if ( v19 == -8 && !v21 )
            v21 = v7;
          v18 = v16 & (v20 + v18);
          v7 = (_QWORD *)(v17 + 16LL * v18);
          v19 = *v7;
          if ( *v7 == a2 )
            goto LABEL_13;
          ++v20;
        }
        if ( v21 )
          v7 = v21;
      }
      goto LABEL_13;
    }
    goto LABEL_41;
  }
  if ( v4 - *(_DWORD *)(a1 + 244) - v13 <= v4 >> 3 )
  {
    sub_A59070(v2, v4);
    v22 = *(_DWORD *)(a1 + 248);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 232);
      v25 = 0;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 1;
      v13 = *(_DWORD *)(a1 + 240) + 1;
      v7 = (_QWORD *)(v24 + 16LL * v26);
      v28 = *v7;
      if ( a2 != *v7 )
      {
        while ( v28 != -4 )
        {
          if ( !v25 && v28 == -8 )
            v25 = v7;
          v26 = v23 & (v27 + v26);
          v7 = (_QWORD *)(v24 + 16LL * v26);
          v28 = *v7;
          if ( a2 == *v7 )
            goto LABEL_13;
          ++v27;
        }
        if ( v25 )
          v7 = v25;
      }
      goto LABEL_13;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 240);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 240) = v13;
  if ( *v7 != -4 )
    --*(_DWORD *)(a1 + 244);
  *v7 = a2;
  v14 = *(_DWORD *)(a1 + 256);
  *((_DWORD *)v7 + 2) = v14;
  result = (unsigned int)(v14 + 1);
  *(_DWORD *)(a1 + 256) = result;
  return result;
}
