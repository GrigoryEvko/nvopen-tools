// Function: sub_25E21D0
// Address: 0x25e21d0
//
_QWORD *__fastcall sub_25E21D0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  _QWORD *result; // rax
  _QWORD *v8; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v8 = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= 3 * v4 )
  {
    v4 *= 2;
  }
  else if ( v4 - *(_DWORD *)(a1 + 20) - v6 > v4 >> 3 )
  {
    goto LABEL_3;
  }
  sub_ED5710(a1, v4);
  sub_25E0B00(a1, a2, &v8);
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v6;
  result = v8;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  return result;
}
