// Function: sub_1674160
// Address: 0x1674160
//
__int64 __fastcall sub_1674160(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // ecx
  _QWORD *v7; // rdx
  __int64 result; // rax
  int v9; // r10d
  _QWORD *v10; // r9
  int v11; // eax
  int v12; // edx
  int v13; // eax
  int v14; // ecx
  __int64 v15; // rdi
  __int64 v16; // rsi
  int v17; // r10d
  _QWORD *v18; // r8
  int v19; // eax
  __int64 v20; // rsi
  int v21; // r8d
  _QWORD *v22; // rdi
  unsigned int v23; // r12d
  __int64 v24; // rcx

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (_QWORD *)(v5 + 8LL * v6);
  result = *v7;
  if ( *v7 == a2 )
    return result;
  v9 = 1;
  v10 = 0;
  while ( result != -8 )
  {
    if ( v10 || result != -16 )
      v7 = v10;
    v6 = (v4 - 1) & (v9 + v6);
    result = *(_QWORD *)(v5 + 8LL * v6);
    if ( result == a2 )
      return result;
    ++v9;
    v10 = v7;
    v7 = (_QWORD *)(v5 + 8LL * v6);
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v10 )
    v10 = v7;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v4 )
  {
LABEL_14:
    sub_1673FB0(a1, 2 * v4);
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      result = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (_QWORD *)(v15 + 8 * result);
      v16 = *v10;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v10 != a2 )
      {
        v17 = 1;
        v18 = 0;
        while ( v16 != -8 )
        {
          if ( v16 == -16 && !v18 )
            v18 = v10;
          result = v14 & (unsigned int)(v17 + result);
          v10 = (_QWORD *)(v15 + 8LL * (unsigned int)result);
          v16 = *v10;
          if ( *v10 == a2 )
            goto LABEL_10;
          ++v17;
        }
        if ( v18 )
          v10 = v18;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  result = v4 - *(_DWORD *)(a1 + 20) - v12;
  if ( (unsigned int)result <= v4 >> 3 )
  {
    sub_1673FB0(a1, v4);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      result = (unsigned int)(v19 - 1);
      v20 = *(_QWORD *)(a1 + 8);
      v21 = 1;
      v22 = 0;
      v23 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (_QWORD *)(v20 + 8LL * v23);
      v24 = *v10;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v10 != a2 )
      {
        while ( v24 != -8 )
        {
          if ( !v22 && v24 == -16 )
            v22 = v10;
          v23 = result & (v21 + v23);
          v10 = (_QWORD *)(v20 + 8LL * v23);
          v24 = *v10;
          if ( *v10 == a2 )
            goto LABEL_10;
          ++v21;
        }
        if ( v22 )
          v10 = v22;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v10 = a2;
  return result;
}
