// Function: sub_1B59B80
// Address: 0x1b59b80
//
__int64 *__fastcall sub_1B59B80(__int64 a1, __int64 *a2)
{
  char v4; // cl
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edx
  __int64 *result; // rax
  __int64 v9; // r9
  unsigned int v10; // esi
  __int64 v11; // rdx
  unsigned int v12; // edi
  unsigned int v13; // r8d
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r11d
  __int64 *v17; // r10
  __int64 *v18; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v10 )
    {
      v11 = *(unsigned int *)(a1 + 8);
      ++*(_QWORD *)a1;
      result = 0;
      v12 = ((unsigned int)v11 >> 1) + 1;
LABEL_8:
      v13 = 3 * v10;
      goto LABEL_9;
    }
    v6 = v10 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = (__int64 *)(v5 + 88LL * v7);
  v9 = *result;
  if ( *a2 == *result )
    return result;
  v16 = 1;
  v17 = 0;
  while ( v9 != -8 )
  {
    if ( !v17 && v9 == -16 )
      v17 = result;
    v7 = v6 & (v16 + v7);
    result = (__int64 *)(v5 + 88LL * v7);
    v9 = *result;
    if ( *a2 == *result )
      return result;
    ++v16;
  }
  v11 = *(unsigned int *)(a1 + 8);
  v13 = 12;
  v10 = 4;
  if ( v17 )
    result = v17;
  ++*(_QWORD *)a1;
  v12 = ((unsigned int)v11 >> 1) + 1;
  if ( !v4 )
  {
    v10 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  v14 = 4 * v12;
  if ( (unsigned int)v14 >= v13 )
  {
    v10 *= 2;
    goto LABEL_15;
  }
  v14 = v10 - *(_DWORD *)(a1 + 12) - v12;
  if ( (unsigned int)v14 <= v10 >> 3 )
  {
LABEL_15:
    sub_1B59660(a1, v10, v11, v14, v13);
    sub_1B50770(a1, a2, &v18);
    result = v18;
    LODWORD(v11) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * ((unsigned int)v11 >> 1) + 2) | v11 & 1;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 12);
  v15 = *a2;
  result[2] = 0x400000000LL;
  *result = v15;
  result[1] = (__int64)(result + 3);
  return result;
}
