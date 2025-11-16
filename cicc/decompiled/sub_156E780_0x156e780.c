// Function: sub_156E780
// Address: 0x156e780
//
_DWORD *__fastcall sub_156E780(__int64 a1, unsigned __int64 a2, int *a3)
{
  unsigned __int64 v4; // rax
  _DWORD *result; // rax
  int v6; // ecx
  _DWORD *i; // rdx

  v4 = *(unsigned int *)(a1 + 12);
  *(_DWORD *)(a1 + 8) = 0;
  if ( a2 > v4 )
    sub_16CD150(a1, a1 + 16, a2, 4);
  result = *(_DWORD **)a1;
  *(_DWORD *)(a1 + 8) = a2;
  v6 = *a3;
  for ( i = &result[(unsigned int)a2]; i != result; ++result )
    *result = v6;
  return result;
}
