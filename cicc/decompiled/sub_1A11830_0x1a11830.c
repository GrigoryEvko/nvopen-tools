// Function: sub_1A11830
// Address: 0x1a11830
//
unsigned __int64 __fastcall sub_1A11830(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  int v5; // r12d
  unsigned int v6; // r14d
  int v7; // r8d
  int v8; // r9d
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // r11d
  __int64 *v17; // r10
  int v18; // edi
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // rdi
  int v24; // r10d
  int v25; // eax
  int v26; // esi
  unsigned int v27; // r12d
  __int64 *v28; // rdi
  __int64 v29; // rdx

  result = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 13 )
  {
    v5 = *(_DWORD *)(result + 12);
    v6 = 0;
    if ( v5 )
    {
      do
      {
        result = (unsigned __int64)sub_1A11440(a1, a2, v6);
        if ( (((unsigned __int8)*(_QWORD *)result ^ 6) & 6) != 0 )
        {
          *(_QWORD *)result |= 6uLL;
          result = *(unsigned int *)(a1 + 824);
          if ( (unsigned int)result >= *(_DWORD *)(a1 + 828) )
          {
            sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, v7, v8);
            result = *(unsigned int *)(a1 + 824);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * result) = a2;
          ++*(_DWORD *)(a1 + 824);
        }
        ++v6;
      }
      while ( v6 != v5 );
    }
    return result;
  }
  v9 = *(_DWORD *)(a1 + 144);
  v10 = a1 + 120;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_26;
  }
  LODWORD(v11) = v9 - 1;
  v12 = *(_QWORD *)(a1 + 128);
  v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v12 + 16LL * v13;
  v14 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
  {
LABEL_11:
    if ( (((unsigned __int8)*(_QWORD *)(result + 8) ^ 6) & 6) == 0 )
      return result;
    v15 = *(_QWORD *)(result + 8) | 6LL;
    goto LABEL_13;
  }
  v16 = 1;
  v17 = 0;
  while ( v14 != -8 )
  {
    if ( v14 == -16 && !v17 )
      v17 = (__int64 *)result;
    v13 = v11 & (v16 + v13);
    result = v12 + 16LL * v13;
    v14 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
      goto LABEL_11;
    ++v16;
  }
  v18 = *(_DWORD *)(a1 + 136);
  if ( v17 )
    result = (unsigned __int64)v17;
  ++*(_QWORD *)(a1 + 120);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_26:
    sub_1A0FE70(v10, 2 * v9);
    v20 = *(_DWORD *)(a1 + 144);
    if ( v20 )
    {
      v21 = v20 - 1;
      v11 = *(_QWORD *)(a1 + 128);
      v22 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 136) + 1;
      result = v11 + 16LL * v22;
      v23 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        v24 = 1;
        v10 = 0;
        while ( v23 != -8 )
        {
          if ( !v10 && v23 == -16 )
            v10 = result;
          v22 = v21 & (v24 + v22);
          result = v11 + 16LL * v22;
          v23 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_22;
          ++v24;
        }
        if ( v10 )
          result = v10;
      }
      goto LABEL_22;
    }
    goto LABEL_54;
  }
  if ( v9 - *(_DWORD *)(a1 + 140) - v19 <= v9 >> 3 )
  {
    sub_1A0FE70(v10, v9);
    v25 = *(_DWORD *)(a1 + 144);
    if ( v25 )
    {
      v26 = v25 - 1;
      v11 = *(_QWORD *)(a1 + 128);
      LODWORD(v10) = 1;
      v27 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 136) + 1;
      v28 = 0;
      result = v11 + 16LL * v27;
      v29 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v28 )
            v28 = (__int64 *)result;
          v27 = v26 & (v10 + v27);
          result = v11 + 16LL * v27;
          v29 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_22;
          LODWORD(v10) = v10 + 1;
        }
        if ( v28 )
          result = (unsigned __int64)v28;
      }
      goto LABEL_22;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_22:
  *(_DWORD *)(a1 + 136) = v19;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 140);
  *(_QWORD *)result = a2;
  v15 = 6;
  *(_QWORD *)(result + 8) = 0;
LABEL_13:
  *(_QWORD *)(result + 8) = v15;
  result = *(unsigned int *)(a1 + 824);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 828) )
  {
    sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, v10, v11);
    result = *(unsigned int *)(a1 + 824);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 824);
  return result;
}
