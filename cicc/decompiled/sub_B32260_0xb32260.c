// Function: sub_B32260
// Address: 0xb32260
//
__int64 __fastcall sub_B32260(__int64 a1, __int64 a2)
{
  char v4; // al
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r10d
  __int64 *v9; // r9
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // rax
  unsigned __int16 v13; // ax
  __int64 v14; // rbx
  __int64 v15; // rax
  int v16; // eax
  int v17; // eax
  int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 v22; // rsi
  int v23; // r10d
  __int64 *v24; // rdi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  int v28; // r8d
  __int64 *v29; // rsi
  unsigned int v30; // ebx
  __int64 v31; // rdx

  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *(_BYTE *)a1;
      if ( (unsigned __int8)(*(_BYTE *)a1 - 2) <= 1u || !v4 )
        return a1;
      if ( v4 != 1 )
        break;
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v6 )
      {
        ++*(_QWORD *)a2;
LABEL_35:
        sub_B32090(a2, 2 * v6);
        v18 = *(_DWORD *)(a2 + 24);
        if ( !v18 )
          goto LABEL_58;
        v19 = v18 - 1;
        v20 = *(_QWORD *)(a2 + 8);
        v21 = (v18 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v9 = (__int64 *)(v20 + 8LL * v21);
        v22 = *v9;
        v17 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v9 != a1 )
        {
          v23 = 1;
          v24 = 0;
          while ( v22 != -4096 )
          {
            if ( !v24 && v22 == -8192 )
              v24 = v9;
            v21 = v19 & (v23 + v21);
            v9 = (__int64 *)(v20 + 8LL * v21);
            v22 = *v9;
            if ( *v9 == a1 )
              goto LABEL_31;
            ++v23;
          }
          if ( v24 )
            v9 = v24;
        }
        goto LABEL_31;
      }
      v7 = *(_QWORD *)(a2 + 8);
      v8 = 1;
      v9 = 0;
      v10 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v11 = (__int64 *)(v7 + 8LL * v10);
      v12 = *v11;
      if ( *v11 == a1 )
        break;
      while ( v12 != -4096 )
      {
        if ( !v9 && v12 == -8192 )
          v9 = v11;
        v10 = (v6 - 1) & (v8 + v10);
        v11 = (__int64 *)(v7 + 8LL * v10);
        v12 = *v11;
        if ( *v11 == a1 )
          goto LABEL_7;
        ++v8;
      }
      v16 = *(_DWORD *)(a2 + 16);
      if ( !v9 )
        v9 = v11;
      ++*(_QWORD *)a2;
      v17 = v16 + 1;
      if ( 4 * v17 >= 3 * v6 )
        goto LABEL_35;
      if ( v6 - (v17 + *(_DWORD *)(a2 + 20)) <= v6 >> 3 )
      {
        sub_B32090(a2, v6);
        v25 = *(_DWORD *)(a2 + 24);
        if ( !v25 )
        {
LABEL_58:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a2 + 8);
        v28 = 1;
        v29 = 0;
        v30 = (v25 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v9 = (__int64 *)(v27 + 8LL * v30);
        v31 = *v9;
        v17 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v9 != a1 )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v29 )
              v29 = v9;
            v30 = v26 & (v28 + v30);
            v9 = (__int64 *)(v27 + 8LL * v30);
            v31 = *v9;
            if ( *v9 == a1 )
              goto LABEL_31;
            ++v28;
          }
          if ( v29 )
            v9 = v29;
        }
      }
LABEL_31:
      *(_DWORD *)(a2 + 16) = v17;
      if ( *v9 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v9 = a1;
      a1 = *(_QWORD *)(a1 - 32);
    }
LABEL_7:
    if ( *(_BYTE *)a1 != 5 )
      return 0;
    v13 = *(_WORD *)(a1 + 2);
    if ( v13 == 15 )
    {
      if ( sub_B32260(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), a2) )
        return 0;
      goto LABEL_20;
    }
    if ( v13 <= 0xFu )
      break;
    if ( v13 != 34 && (unsigned __int16)(v13 - 47) > 2u )
      return 0;
LABEL_20:
    a1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  }
  if ( v13 == 13 )
  {
    v14 = sub_B32260(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), a2);
    v15 = sub_B32260(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), a2);
    a1 = v15;
    if ( !v14 || !v15 )
    {
      if ( v14 )
        return v14;
      return a1;
    }
  }
  return 0;
}
