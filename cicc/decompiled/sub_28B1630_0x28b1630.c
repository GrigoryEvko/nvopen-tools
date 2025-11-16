// Function: sub_28B1630
// Address: 0x28b1630
//
__int64 __fastcall sub_28B1630(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rdx
  __int64 v9; // rax
  int v10; // r10d
  _QWORD *v11; // r9
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rsi
  int v19; // r10d
  _QWORD *v20; // r8
  int v21; // eax
  int v22; // eax
  __int64 v23; // rdi
  int v24; // r8d
  _QWORD *v25; // rsi
  unsigned int v26; // r13d
  __int64 v27; // rcx

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 1;
  result = 1;
  if ( *(_QWORD *)(**(_QWORD **)a1 + 40LL) != *(_QWORD *)(a2 + 40) )
    return result;
  if ( **(_QWORD **)(a1 + 8) != a2 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v5 = *(_DWORD *)(v4 + 24);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v4 + 8);
      v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (_QWORD *)(v6 + 8LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        return 1;
      v10 = 1;
      v11 = 0;
      while ( v9 != -4096 )
      {
        if ( !v11 && v9 == -8192 )
          v11 = v8;
        v7 = (v5 - 1) & (v10 + v7);
        v8 = (_QWORD *)(v6 + 8LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          return 1;
        ++v10;
      }
      v12 = *(_DWORD *)(v4 + 16);
      if ( !v11 )
        v11 = v8;
      ++*(_QWORD *)v4;
      v13 = v12 + 1;
      if ( 4 * (v12 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(v4 + 20) - v13 > v5 >> 3 )
        {
LABEL_15:
          *(_DWORD *)(v4 + 16) = v13;
          if ( *v11 != -4096 )
            --*(_DWORD *)(v4 + 20);
          *v11 = a2;
          return 1;
        }
        sub_CF4090(v4, v5);
        v21 = *(_DWORD *)(v4 + 24);
        if ( v21 )
        {
          v22 = v21 - 1;
          v23 = *(_QWORD *)(v4 + 8);
          v24 = 1;
          v25 = 0;
          v26 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v11 = (_QWORD *)(v23 + 8LL * v26);
          v27 = *v11;
          v13 = *(_DWORD *)(v4 + 16) + 1;
          if ( a2 != *v11 )
          {
            while ( v27 != -4096 )
            {
              if ( v27 == -8192 && !v25 )
                v25 = v11;
              v26 = v22 & (v24 + v26);
              v11 = (_QWORD *)(v23 + 8LL * v26);
              v27 = *v11;
              if ( a2 == *v11 )
                goto LABEL_15;
              ++v24;
            }
            if ( v25 )
              v11 = v25;
          }
          goto LABEL_15;
        }
LABEL_47:
        ++*(_DWORD *)(v4 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v4;
    }
    sub_CF4090(v4, 2 * v5);
    v14 = *(_DWORD *)(v4 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v4 + 8);
      v17 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (_QWORD *)(v16 + 8LL * v17);
      v18 = *v11;
      v13 = *(_DWORD *)(v4 + 16) + 1;
      if ( a2 != *v11 )
      {
        v19 = 1;
        v20 = 0;
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v20 )
            v20 = v11;
          v17 = v15 & (v19 + v17);
          v11 = (_QWORD *)(v16 + 8LL * v17);
          v18 = *v11;
          if ( a2 == *v11 )
            goto LABEL_15;
          ++v19;
        }
        if ( v20 )
          v11 = v20;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  return 0;
}
