// Function: sub_1DD76A0
// Address: 0x1dd76a0
//
_DWORD *__fastcall sub_1DD76A0(__int64 a1, __int64 a2, int a3)
{
  _DWORD *result; // rax

  result = *(_DWORD **)(a1 + 120);
  if ( *(_DWORD **)(a1 + 112) != result )
  {
    result = (_DWORD *)sub_1DD7680(a1, a2);
    *result = a3;
  }
  return result;
}
