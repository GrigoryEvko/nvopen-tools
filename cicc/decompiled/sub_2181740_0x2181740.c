// Function: sub_2181740
// Address: 0x2181740
//
int *__fastcall sub_2181740(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // ecx
  int *result; // rax
  int v8; // edi
  int v9; // r11d
  int *v10; // r10
  int v11; // ecx
  int v12; // ecx
  int v13; // edx
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
  v6 = (v4 - 1) & (37 * *a2);
  result = (int *)(v5 + 40LL * v6);
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
    result = (int *)(v5 + 40LL * v6);
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
    sub_1DF5170(a1, v4);
    sub_217F350(a1, a2, v14);
    result = (int *)v14[0];
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v12;
  if ( *result != -1 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  *((_QWORD *)result + 2) = 0x400000000LL;
  *result = v13;
  *((_QWORD *)result + 1) = result + 6;
  return result;
}
