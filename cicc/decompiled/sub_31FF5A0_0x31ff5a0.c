// Function: sub_31FF5A0
// Address: 0x31ff5a0
//
unsigned __int64 __fastcall sub_31FF5A0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v5; // esi
  __int64 v6; // r10
  __int64 v7; // r8
  unsigned int v8; // r13d
  unsigned int v9; // edi
  unsigned __int64 result; // rax
  __int64 v11; // rcx
  int v12; // r11d
  _QWORD *v13; // rdx
  int v14; // eax
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // r9d
  _QWORD *v24; // r8
  __int64 v25; // rdi
  unsigned int v26; // r13d
  __int64 v27; // rsi

  v3 = *a1;
  v5 = *(_DWORD *)(*a1 + 456);
  v6 = *a1 + 432;
  if ( !v5 )
  {
    ++*(_QWORD *)(v3 + 432);
    goto LABEL_14;
  }
  v7 = *(_QWORD *)(v3 + 440);
  v8 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v9 = (v5 - 1) & v8;
  result = v7 + 16LL * v9;
  v11 = *(_QWORD *)result;
  if ( a3 == *(_QWORD *)result )
    return result;
  v12 = 1;
  v13 = 0;
  while ( v11 != -4096 )
  {
    if ( v11 != -8192 || v13 )
      result = (unsigned __int64)v13;
    v9 = (v5 - 1) & (v12 + v9);
    v11 = *(_QWORD *)(v7 + 16LL * v9);
    if ( a3 == v11 )
      return result;
    ++v12;
    v13 = (_QWORD *)result;
    result = v7 + 16LL * v9;
  }
  if ( !v13 )
    v13 = (_QWORD *)result;
  v14 = *(_DWORD *)(v3 + 448);
  ++*(_QWORD *)(v3 + 432);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_14:
    sub_31FF3C0(v6, 2 * v5);
    v16 = *(_DWORD *)(v3 + 456);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v3 + 440);
      result = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v15 = *(_DWORD *)(v3 + 448) + 1;
      v13 = (_QWORD *)(v18 + 16 * result);
      v19 = *v13;
      if ( a3 != *v13 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4096 )
        {
          if ( v19 == -8192 && !v21 )
            v21 = v13;
          result = v17 & (unsigned int)(v20 + result);
          v13 = (_QWORD *)(v18 + 16LL * (unsigned int)result);
          v19 = *v13;
          if ( a3 == *v13 )
            goto LABEL_10;
          ++v20;
        }
        if ( v21 )
          v13 = v21;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  result = v5 - *(_DWORD *)(v3 + 452) - v15;
  if ( (unsigned int)result <= v5 >> 3 )
  {
    sub_31FF3C0(v6, v5);
    v22 = *(_DWORD *)(v3 + 456);
    if ( v22 )
    {
      result = (unsigned int)(v22 - 1);
      v23 = 1;
      v24 = 0;
      v25 = *(_QWORD *)(v3 + 440);
      v26 = result & v8;
      v15 = *(_DWORD *)(v3 + 448) + 1;
      v13 = (_QWORD *)(v25 + 16LL * v26);
      v27 = *v13;
      if ( a3 != *v13 )
      {
        while ( v27 != -4096 )
        {
          if ( !v24 && v27 == -8192 )
            v24 = v13;
          v26 = result & (v23 + v26);
          v13 = (_QWORD *)(v25 + 16LL * v26);
          v27 = *v13;
          if ( a3 == *v13 )
            goto LABEL_10;
          ++v23;
        }
        if ( v24 )
          v13 = v24;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(v3 + 448);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(v3 + 448) = v15;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v3 + 452);
  *v13 = a3;
  v13[1] = 0;
  return result;
}
