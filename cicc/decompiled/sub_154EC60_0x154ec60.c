// Function: sub_154EC60
// Address: 0x154ec60
//
__int64 *__fastcall sub_154EC60(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // rdi
  unsigned int v7; // r9d
  unsigned int v8; // ecx
  __int64 *result; // rax
  __int64 v10; // rdx
  __int64 *v11; // r10
  int v12; // r13d
  int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int v16; // r11d
  int v17; // r10d
  int v18; // r11d
  __int64 *v19; // r10
  int v20; // edi
  int v21; // r13d
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v23[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 152;
  v22 = a2;
  v4 = *(_DWORD *)(a1 + 176);
  v5 = *(_QWORD *)(a1 + 160);
  if ( !v4 )
  {
    v12 = *(_DWORD *)(a1 + 184);
    ++*(_QWORD *)(a1 + 152);
    *(_DWORD *)(a1 + 184) = v12 + 1;
    goto LABEL_6;
  }
  v6 = v22;
  v7 = v4 - 1;
  v8 = (v4 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  result = (__int64 *)(v5 + 16LL * v8);
  v10 = *result;
  v11 = result;
  if ( v22 != *result )
  {
    v16 = (v4 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v17 = 1;
    while ( v10 != -4 )
    {
      v21 = v17 + 1;
      v16 = v7 & (v16 + v17);
      v11 = (__int64 *)(v5 + 16LL * v16);
      v10 = *v11;
      if ( v22 == *v11 )
        goto LABEL_3;
      v17 = v21;
    }
    v12 = *(_DWORD *)(a1 + 184);
    *(_DWORD *)(a1 + 184) = v12 + 1;
    v8 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    result = (__int64 *)(v5 + 16LL * v8);
LABEL_12:
    v15 = *result;
    if ( *result == v6 )
    {
LABEL_13:
      *((_DWORD *)result + 2) = v12;
      return result;
    }
    v18 = 1;
    v19 = 0;
    while ( v15 != -4 )
    {
      if ( v15 == -8 && !v19 )
        v19 = result;
      v8 = v7 & (v18 + v8);
      result = (__int64 *)(v5 + 16LL * v8);
      v15 = *result;
      if ( *result == v6 )
        goto LABEL_13;
      ++v18;
    }
    v20 = *(_DWORD *)(a1 + 168);
    if ( v19 )
      result = v19;
    ++*(_QWORD *)(a1 + 152);
    v13 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 172) - v13 > v4 >> 3 )
        goto LABEL_8;
      goto LABEL_7;
    }
LABEL_6:
    v4 *= 2;
LABEL_7:
    sub_154EA90(v2, v4);
    sub_154CDE0(v2, &v22, v23);
    result = (__int64 *)v23[0];
    v13 = *(_DWORD *)(a1 + 168) + 1;
LABEL_8:
    *(_DWORD *)(a1 + 168) = v13;
    if ( *result != -4 )
      --*(_DWORD *)(a1 + 172);
    v14 = v22;
    *((_DWORD *)result + 2) = 0;
    *result = v14;
    goto LABEL_13;
  }
LABEL_3:
  if ( v11 == (__int64 *)(v5 + 16LL * v4) )
  {
    v12 = *(_DWORD *)(a1 + 184);
    *(_DWORD *)(a1 + 184) = v12 + 1;
    goto LABEL_12;
  }
  return result;
}
