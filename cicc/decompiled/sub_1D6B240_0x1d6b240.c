// Function: sub_1D6B240
// Address: 0x1d6b240
//
__int64 *__fastcall sub_1D6B240(__int64 a1, __int64 *a2)
{
  char v3; // r8
  __int64 *result; // rax
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  __int64 v8; // rdx
  __int64 *v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_1D66970(a1, a2, v9);
  result = v9[0];
  if ( !v3 )
  {
    v5 = *(_DWORD *)(a1 + 16);
    v6 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)a1;
    v7 = v5 + 1;
    if ( 4 * (v5 + 1) >= 3 * v6 )
    {
      sub_1D6AF90(a1, 2 * v6);
      sub_1D66970(a1, a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    else if ( v6 - *(_DWORD *)(a1 + 20) - v7 <= v6 >> 3 )
    {
      sub_1D6AF90(a1, v6);
      sub_1D66970(a1, a2, v9);
      result = v9[0];
      v7 = *(_DWORD *)(a1 + 16) + 1;
    }
    *(_DWORD *)(a1 + 16) = v7;
    if ( *result != -8 || result[1] != -8 )
      --*(_DWORD *)(a1 + 20);
    *result = *a2;
    v8 = a2[1];
    result[2] = 0;
    result[1] = v8;
  }
  return result;
}
