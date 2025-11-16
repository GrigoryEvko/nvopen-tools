// Function: sub_37C5CE0
// Address: 0x37c5ce0
//
__int16 *__fastcall sub_37C5CE0(__int64 a1, unsigned __int16 *a2, __int16 *a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  __int16 *result; // rax
  __int16 *v8; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v8 = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= 3 * v4 )
  {
    v4 *= 2;
    goto LABEL_8;
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v6 <= v4 >> 3 )
  {
LABEL_8:
    sub_37C5A70(a1, v4);
    sub_37BD660(a1, a2, &v8);
    v6 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v6;
  result = v8;
  if ( *v8 != -1 || v8[1] != -1 )
    --*(_DWORD *)(a1 + 20);
  return result;
}
