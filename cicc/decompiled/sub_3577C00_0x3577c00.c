// Function: sub_3577C00
// Address: 0x3577c00
//
__int64 __fastcall sub_3577C00(__int64 a1, int a2)
{
  __int64 v2; // r10
  unsigned int v4; // esi
  __int64 v6; // r8
  unsigned int v7; // edi
  _DWORD *v8; // rcx
  __int64 result; // rax
  int v10; // edx
  int v11; // eax
  _DWORD *v12; // r11
  int v13; // eax
  int v14; // edx
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  int v19; // esi
  int v20; // r9d
  _DWORD *v21; // r8
  int v22; // eax
  int v23; // eax
  __int64 v24; // rsi
  int v25; // r8d
  _DWORD *v26; // rdi
  unsigned int v27; // r13d
  int v28; // ecx

  v2 = a1 + 240;
  v4 = *(_DWORD *)(a1 + 264);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_13;
  }
  v6 = *(_QWORD *)(a1 + 248);
  v7 = (v4 - 1) & (37 * a2);
  v8 = (_DWORD *)(v6 + 4LL * v7);
  result = 0;
  v10 = *v8;
  if ( a2 == *v8 )
    return result;
  v11 = 1;
  v12 = 0;
  while ( v10 != -1 )
  {
    if ( v12 || v10 != -2 )
      v8 = v12;
    v7 = (v4 - 1) & (v11 + v7);
    v10 = *(_DWORD *)(v6 + 4LL * v7);
    if ( v10 == a2 )
      return 0;
    ++v11;
    v12 = v8;
    v8 = (_DWORD *)(v6 + 4LL * v7);
  }
  v13 = *(_DWORD *)(a1 + 256);
  if ( !v12 )
    v12 = v8;
  ++*(_QWORD *)(a1 + 240);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_13:
    sub_2E29BA0(v2, 2 * v4);
    v15 = *(_DWORD *)(a1 + 264);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 248);
      v18 = (v15 - 1) & (37 * a2);
      v12 = (_DWORD *)(v17 + 4LL * v18);
      v19 = *v12;
      v14 = *(_DWORD *)(a1 + 256) + 1;
      if ( *v12 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -1 )
        {
          if ( v19 == -2 && !v21 )
            v21 = v12;
          v18 = v16 & (v20 + v18);
          v12 = (_DWORD *)(v17 + 4LL * v18);
          v19 = *v12;
          if ( *v12 == a2 )
            goto LABEL_9;
          ++v20;
        }
        if ( v21 )
          v12 = v21;
      }
      goto LABEL_9;
    }
    goto LABEL_42;
  }
  if ( v4 - *(_DWORD *)(a1 + 260) - v14 <= v4 >> 3 )
  {
    sub_2E29BA0(v2, v4);
    v22 = *(_DWORD *)(a1 + 264);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 248);
      v25 = 1;
      v26 = 0;
      v27 = v23 & (37 * a2);
      v12 = (_DWORD *)(v24 + 4LL * v27);
      v28 = *v12;
      v14 = *(_DWORD *)(a1 + 256) + 1;
      if ( *v12 != a2 )
      {
        while ( v28 != -1 )
        {
          if ( v28 == -2 && !v26 )
            v26 = v12;
          v27 = v23 & (v25 + v27);
          v12 = (_DWORD *)(v24 + 4LL * v27);
          v28 = *v12;
          if ( a2 == *v12 )
            goto LABEL_9;
          ++v25;
        }
        if ( v26 )
          v12 = v26;
      }
      goto LABEL_9;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 256);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(a1 + 256) = v14;
  if ( *v12 != -1 )
    --*(_DWORD *)(a1 + 260);
  *v12 = a2;
  return 1;
}
