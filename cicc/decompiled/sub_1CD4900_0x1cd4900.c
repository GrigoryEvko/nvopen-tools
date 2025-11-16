// Function: sub_1CD4900
// Address: 0x1cd4900
//
__int64 *__fastcall sub_1CD4900(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *result; // rax
  __int64 v8; // rdi
  int v9; // r11d
  __int64 *v10; // r10
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rdx
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_14:
    v4 *= 2;
    goto LABEL_15;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = (__int64 *)(v5 + 16LL * v6);
  v8 = *result;
  if ( *a2 == *result )
    return result;
  v9 = 1;
  v10 = 0;
  while ( v8 != -8 )
  {
    if ( v8 == -16 && !v10 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (__int64 *)(v5 + 16LL * v6);
    v8 = *result;
    if ( *a2 == *result )
      return result;
    ++v9;
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( v10 )
    result = v10;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v4 )
    goto LABEL_14;
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
LABEL_15:
    sub_18EEB70(a1, v4);
    sub_1CD3300(a1, a2, v14);
    result = (__int64 *)v14[0];
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  *((_DWORD *)result + 2) = 0;
  *result = v13;
  return result;
}
