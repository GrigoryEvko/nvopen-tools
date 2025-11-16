// Function: sub_1A10C00
// Address: 0x1a10c00
//
unsigned __int64 __fastcall sub_1A10C00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  _QWORD *v8; // r9
  _QWORD *v9; // r8
  unsigned __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // r11d
  _QWORD *v16; // rcx
  int v17; // eax
  int v18; // edx
  __int64 v19; // rdx
  int v20; // eax
  int v21; // esi
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r10d
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // r14d
  __int64 v29; // rsi

  v6 = a1 + 120;
  v7 = *(_DWORD *)(a1 + 144);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_27;
  }
  v8 = *(_QWORD **)(a1 + 128);
  LODWORD(v9) = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (unsigned __int64)&v8[2 * (unsigned int)v9];
  v11 = *(_QWORD *)result;
  if ( *(_QWORD *)result != a2 )
  {
    v15 = 1;
    v16 = 0;
    while ( v11 != -8 )
    {
      if ( !v16 && v11 == -16 )
        v16 = (_QWORD *)result;
      LODWORD(v9) = (v7 - 1) & (v15 + (_DWORD)v9);
      result = (unsigned __int64)&v8[2 * (unsigned int)v9];
      v11 = *(_QWORD *)result;
      if ( *(_QWORD *)result == a2 )
        goto LABEL_3;
      ++v15;
    }
    if ( !v16 )
      v16 = (_QWORD *)result;
    v17 = *(_DWORD *)(a1 + 136);
    ++*(_QWORD *)(a1 + 120);
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v7 )
    {
      LODWORD(v9) = v7 >> 3;
      if ( v7 - *(_DWORD *)(a1 + 140) - v18 > v7 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 136) = v18;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 140);
        *v16 = a2;
        v19 = 0;
        v16[1] = 0;
        goto LABEL_25;
      }
      sub_1A0FE70(v6, v7);
      v25 = *(_DWORD *)(a1 + 144);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 128);
        v9 = 0;
        v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        LODWORD(v8) = 1;
        v18 = *(_DWORD *)(a1 + 136) + 1;
        v16 = (_QWORD *)(v27 + 16LL * v28);
        v29 = *v16;
        if ( *v16 != a2 )
        {
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v9 )
              v9 = v16;
            v28 = v26 & ((_DWORD)v8 + v28);
            v16 = (_QWORD *)(v27 + 16LL * v28);
            v29 = *v16;
            if ( *v16 == a2 )
              goto LABEL_22;
            LODWORD(v8) = (_DWORD)v8 + 1;
          }
          if ( v9 )
            v16 = v9;
        }
        goto LABEL_22;
      }
LABEL_56:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_27:
    sub_1A0FE70(v6, 2 * v7);
    v20 = *(_DWORD *)(a1 + 144);
    if ( v20 )
    {
      v21 = v20 - 1;
      v9 = *(_QWORD **)(a1 + 128);
      v22 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 136) + 1;
      v16 = &v9[2 * v22];
      v23 = *v16;
      if ( *v16 != a2 )
      {
        v24 = 1;
        v8 = 0;
        while ( v23 != -8 )
        {
          if ( !v8 && v23 == -16 )
            v8 = v16;
          v22 = v21 & (v24 + v22);
          v16 = &v9[2 * v22];
          v23 = *v16;
          if ( *v16 == a2 )
            goto LABEL_22;
          ++v24;
        }
        if ( v8 )
          v16 = v8;
      }
      goto LABEL_22;
    }
    goto LABEL_56;
  }
LABEL_3:
  v12 = *(_QWORD *)(result + 8);
  v13 = (v12 >> 1) & 3;
  if ( v13 == 3 || v13 == 1 )
    return result;
  if ( (_DWORD)v13 )
  {
    if ( a3 == (v12 & 0xFFFFFFFFFFFFFFF8LL) )
      return result;
    v14 = v12 | 6;
    *(_QWORD *)(result + 8) = v14;
    goto LABEL_8;
  }
  v19 = v12 & 1;
  v16 = (_QWORD *)result;
LABEL_25:
  v14 = a3 | v19 | 2;
  v16[1] = v14;
LABEL_8:
  if ( (((unsigned __int8)v14 ^ 6) & 6) != 0 )
  {
    result = *(unsigned int *)(a1 + 1352);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 1356) )
    {
      sub_16CD150(a1 + 1344, (const void *)(a1 + 1360), 0, 8, (int)v9, (int)v8);
      result = *(unsigned int *)(a1 + 1352);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 1352);
  }
  else
  {
    result = *(unsigned int *)(a1 + 824);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 828) )
    {
      sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, (int)v9, (int)v8);
      result = *(unsigned int *)(a1 + 824);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 824);
  }
  return result;
}
