// Function: sub_1E49390
// Address: 0x1e49390
//
_DWORD *__fastcall sub_1E49390(__int64 a1, int *a2)
{
  char v3; // r8
  _DWORD *result; // rax
  int v5; // ecx
  unsigned int v6; // esi
  int v7; // edx
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_1932870(a1, a2, v8);
  result = (_DWORD *)v8[0];
  if ( v3 )
    return result;
  v5 = *(_DWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)a1;
  v7 = v5 + 1;
  if ( 4 * (v5 + 1) >= 3 * v6 )
  {
    v6 *= 2;
    goto LABEL_8;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v7 <= v6 >> 3 )
  {
LABEL_8:
    sub_1392B70(a1, v6);
    sub_1932870(a1, a2, v8);
    result = (_DWORD *)v8[0];
    v7 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v7;
  if ( *result != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)result = (unsigned int)*a2;
  return result;
}
