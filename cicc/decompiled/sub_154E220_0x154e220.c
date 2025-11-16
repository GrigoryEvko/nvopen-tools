// Function: sub_154E220
// Address: 0x154e220
//
__int64 *__fastcall sub_154E220(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // r12d
  unsigned int v5; // r8d
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *result; // rax
  __int64 v9; // rcx
  int v10; // r11d
  __int64 *v11; // r10
  int v12; // ecx
  int v13; // ecx
  int v14; // esi
  __int64 v15; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 32;
  v4 = *(_DWORD *)(a1 + 64);
  v5 = *(_DWORD *)(a1 + 56);
  v15 = a2;
  *(_DWORD *)(a1 + 64) = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 32);
LABEL_14:
    v14 = 2 * v5;
LABEL_15:
    sub_1542080(v2, v14);
    sub_154CC80(v2, &v15, v16);
    result = (__int64 *)v16[0];
    a2 = v15;
    v13 = *(_DWORD *)(a1 + 48) + 1;
    goto LABEL_10;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v6 + 16LL * v7);
  v9 = *result;
  if ( a2 == *result )
    goto LABEL_3;
  v10 = 1;
  v11 = 0;
  while ( v9 != -8 )
  {
    if ( !v11 && v9 == -16 )
      v11 = result;
    v7 = (v5 - 1) & (v10 + v7);
    result = (__int64 *)(v6 + 16LL * v7);
    v9 = *result;
    if ( a2 == *result )
      goto LABEL_3;
    ++v10;
  }
  v12 = *(_DWORD *)(a1 + 48);
  if ( v11 )
    result = v11;
  ++*(_QWORD *)(a1 + 32);
  v13 = v12 + 1;
  if ( 4 * v13 >= 3 * v5 )
    goto LABEL_14;
  if ( v5 - *(_DWORD *)(a1 + 52) - v13 <= v5 >> 3 )
  {
    v14 = v5;
    goto LABEL_15;
  }
LABEL_10:
  *(_DWORD *)(a1 + 48) = v13;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 52);
  *result = a2;
  *((_DWORD *)result + 2) = 0;
LABEL_3:
  *((_DWORD *)result + 2) = v4;
  return result;
}
