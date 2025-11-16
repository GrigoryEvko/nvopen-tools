// Function: sub_22B3610
// Address: 0x22b3610
//
_DWORD *__fastcall sub_22B3610(__int64 a1, int *a2, _DWORD *a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  _DWORD *result; // rax
  _DWORD *v8; // [rsp+8h] [rbp-18h] BYREF

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
  sub_1247200(a1, v4);
  sub_22B1B10(a1, a2, &v8);
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v6;
  result = v8;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  return result;
}
