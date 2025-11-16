// Function: sub_1A29A10
// Address: 0x1a29a10
//
__int64 __fastcall sub_1A29A10(__int64 a1, __int64 *a2)
{
  char v4; // cl
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edx
  __int64 result; // rax
  __int64 v9; // r9
  unsigned int v10; // esi
  unsigned int v11; // edx
  int v12; // edi
  unsigned int v13; // r8d
  __int64 v14; // rdx
  int v15; // r11d
  __int64 v16; // r10
  __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

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
      v11 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      result = 0;
      v12 = (v11 >> 1) + 1;
LABEL_8:
      v13 = 3 * v10;
      goto LABEL_9;
    }
    v6 = v10 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = v5 + 16LL * v7;
  v9 = *(_QWORD *)result;
  if ( *a2 == *(_QWORD *)result )
    return result;
  v15 = 1;
  v16 = 0;
  while ( v9 != -8 )
  {
    if ( !v16 && v9 == -16 )
      v16 = result;
    v7 = v6 & (v15 + v7);
    result = v5 + 16LL * v7;
    v9 = *(_QWORD *)result;
    if ( *a2 == *(_QWORD *)result )
      return result;
    ++v15;
  }
  v11 = *(_DWORD *)(a1 + 8);
  v13 = 12;
  v10 = 4;
  if ( v16 )
    result = v16;
  ++*(_QWORD *)a1;
  v12 = (v11 >> 1) + 1;
  if ( !v4 )
  {
    v10 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v12 >= v13 )
  {
    v10 *= 2;
    goto LABEL_15;
  }
  if ( v10 - *(_DWORD *)(a1 + 12) - v12 <= v10 >> 3 )
  {
LABEL_15:
    sub_1A29660(a1, v10);
    sub_1A26F00(a1, a2, v17);
    result = v17[0];
    v11 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v11 >> 1) + 2) | v11 & 1;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 12);
  v14 = *a2;
  *(_BYTE *)(result + 8) = 0;
  *(_QWORD *)result = v14;
  return result;
}
