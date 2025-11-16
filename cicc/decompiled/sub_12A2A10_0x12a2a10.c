// Function: sub_12A2A10
// Address: 0x12a2a10
//
__int64 __fastcall sub_12A2A10(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  unsigned int v5; // r8d
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 result; // rax
  int v9; // r11d
  _QWORD *v10; // rdx
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  int v14; // esi
  __int64 v15; // r9
  unsigned int v16; // eax
  __int64 v17; // r8
  int v18; // r11d
  _QWORD *v19; // r10
  int v20; // eax
  int v21; // eax
  __int64 v22; // r8
  int v23; // r10d
  unsigned int v24; // r12d
  _QWORD *v25; // r9
  __int64 v26; // rsi

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_17;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (_QWORD *)(v4 + 16LL * v5);
  v7 = *v6;
  if ( *v6 == a2 )
  {
LABEL_3:
    result = v6[1];
    if ( result )
    {
      if ( *(_DWORD *)(*(_QWORD *)result + 8LL) >> 8 )
        return sub_1289750((_QWORD *)a1, result);
    }
    return result;
  }
  v9 = 1;
  v10 = 0;
  while ( v7 != -8 )
  {
    if ( !v10 && v7 == -16 )
      v10 = v6;
    v5 = (v3 - 1) & (v9 + v5);
    v6 = (_QWORD *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v6;
  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v3 )
  {
LABEL_17:
    sub_12A2850(a1, 2 * v3);
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v15 + 16LL * v16);
      v17 = *v10;
      if ( *v10 != a2 )
      {
        v18 = 1;
        v19 = 0;
        while ( v17 != -8 )
        {
          if ( !v19 && v17 == -16 )
            v19 = v10;
          v16 = v14 & (v18 + v16);
          v10 = (_QWORD *)(v15 + 16LL * v16);
          v17 = *v10;
          if ( *v10 == a2 )
            goto LABEL_13;
          ++v18;
        }
        if ( v19 )
          v10 = v19;
      }
      goto LABEL_13;
    }
    goto LABEL_45;
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v12 <= v3 >> 3 )
  {
    sub_12A2850(a1, v3);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = 1;
      v24 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = 0;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v22 + 16LL * v24);
      v26 = *v10;
      if ( *v10 != a2 )
      {
        while ( v26 != -8 )
        {
          if ( !v25 && v26 == -16 )
            v25 = v10;
          v24 = v21 & (v23 + v24);
          v10 = (_QWORD *)(v22 + 16LL * v24);
          v26 = *v10;
          if ( *v10 == a2 )
            goto LABEL_13;
          ++v23;
        }
        if ( v25 )
          v10 = v25;
      }
      goto LABEL_13;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v10 = a2;
  v10[1] = 0;
  return 0;
}
