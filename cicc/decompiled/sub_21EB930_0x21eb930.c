// Function: sub_21EB930
// Address: 0x21eb930
//
_DWORD *__fastcall sub_21EB930(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // ecx
  _DWORD *result; // rax
  int v8; // edi
  int v9; // r11d
  _DWORD *v10; // r10
  int v11; // ecx
  int v12; // ecx
  _DWORD *v13; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_14:
    v4 *= 2;
    goto LABEL_15;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (37 * *a2);
  result = (_DWORD *)(v5 + 8LL * v6);
  v8 = *result;
  if ( *a2 == *result )
    return result;
  v9 = 1;
  v10 = 0;
  while ( v8 != -1 )
  {
    if ( v8 == -2 && !v10 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (_DWORD *)(v5 + 8LL * v6);
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
    sub_1BFDD60(a1, v4);
    sub_1BFD720(a1, a2, &v13);
    result = v13;
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *result != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)result = (unsigned int)*a2;
  return result;
}
