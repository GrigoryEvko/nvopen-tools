// Function: sub_1F108D0
// Address: 0x1f108d0
//
unsigned __int64 __fastcall sub_1F108D0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned int v3; // ecx
  __int64 v5; // rdi
  __int64 *v6; // rdx
  __int64 v7; // r9
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r13
  __int64 v10; // rdi
  unsigned int v11; // esi
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 v14; // r11
  int v15; // r15d
  unsigned __int64 *v16; // r9
  int v17; // eax
  int v18; // edx
  int v19; // edx
  int v20; // r10d
  int v21; // eax
  int v22; // esi
  __int64 v23; // rdi
  unsigned __int64 v24; // rcx
  int v25; // r10d
  unsigned __int64 *v26; // r8
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  int v30; // r8d
  unsigned int v31; // r14d
  unsigned __int64 *v32; // rsi

  result = *(unsigned int *)(a1 + 384);
  if ( !(_DWORD)result )
    return result;
  v3 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = *(_QWORD *)(a1 + 368);
  v6 = (__int64 *)(v5 + 16LL * v3);
  v7 = *v6;
  if ( a2 == *v6 )
  {
LABEL_3:
    result = v5 + 16 * result;
    if ( v6 == (__int64 *)result )
      return result;
    v8 = v6[1];
    *v6 = -16;
    --*(_DWORD *)(a1 + 376);
    ++*(_DWORD *)(a1 + 380);
    result = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(a2 + 46) & 8) == 0 )
    {
      *(_QWORD *)(result + 16) = 0;
      return result;
    }
    v9 = *(_QWORD *)(a2 + 8);
    v10 = a1 + 360;
    *(_QWORD *)(result + 16) = v9;
    v11 = *(_DWORD *)(a1 + 384);
    if ( v11 )
    {
      v12 = *(_QWORD *)(a1 + 368);
      v13 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      result = v12 + 16LL * v13;
      v14 = *(_QWORD *)result;
      if ( *(_QWORD *)result == v9 )
        return result;
      v15 = 1;
      v16 = 0;
      while ( v14 != -8 )
      {
        if ( !v16 && v14 == -16 )
          v16 = (unsigned __int64 *)result;
        v13 = (v11 - 1) & (v15 + v13);
        result = v12 + 16LL * v13;
        v14 = *(_QWORD *)result;
        if ( *(_QWORD *)result == v9 )
          return result;
        ++v15;
      }
      if ( !v16 )
        v16 = (unsigned __int64 *)result;
      v17 = *(_DWORD *)(a1 + 376);
      ++*(_QWORD *)(a1 + 360);
      v18 = v17 + 1;
      if ( 4 * (v17 + 1) < 3 * v11 )
      {
        result = v11 - *(_DWORD *)(a1 + 380) - v18;
        if ( (unsigned int)result > v11 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a1 + 376) = v18;
          if ( *v16 != -8 )
            --*(_DWORD *)(a1 + 380);
          *v16 = v9;
          v16[1] = v8;
          return result;
        }
        sub_1DC1390(v10, v11);
        v27 = *(_DWORD *)(a1 + 384);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a1 + 368);
          v30 = 1;
          v31 = (v27 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v18 = *(_DWORD *)(a1 + 376) + 1;
          v32 = 0;
          v16 = (unsigned __int64 *)(v29 + 16LL * v31);
          result = *v16;
          if ( *v16 != v9 )
          {
            while ( result != -8 )
            {
              if ( !v32 && result == -16 )
                v32 = v16;
              v31 = v28 & (v30 + v31);
              v16 = (unsigned __int64 *)(v29 + 16LL * v31);
              result = *v16;
              if ( *v16 == v9 )
                goto LABEL_16;
              ++v30;
            }
            if ( v32 )
              v16 = v32;
          }
          goto LABEL_16;
        }
LABEL_52:
        ++*(_DWORD *)(a1 + 376);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 360);
    }
    sub_1DC1390(v10, 2 * v11);
    v21 = *(_DWORD *)(a1 + 384);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 368);
      result = (v21 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v18 = *(_DWORD *)(a1 + 376) + 1;
      v16 = (unsigned __int64 *)(v23 + 16 * result);
      v24 = *v16;
      if ( v9 != *v16 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( !v26 && v24 == -16 )
            v26 = v16;
          result = v22 & (unsigned int)(v25 + result);
          v16 = (unsigned __int64 *)(v23 + 16LL * (unsigned int)result);
          v24 = *v16;
          if ( *v16 == v9 )
            goto LABEL_16;
          ++v25;
        }
        if ( v26 )
          v16 = v26;
      }
      goto LABEL_16;
    }
    goto LABEL_52;
  }
  v19 = 1;
  while ( v7 != -8 )
  {
    v20 = v19 + 1;
    v3 = (result - 1) & (v19 + v3);
    v6 = (__int64 *)(v5 + 16LL * v3);
    v7 = *v6;
    if ( a2 == *v6 )
      goto LABEL_3;
    v19 = v20;
  }
  return result;
}
