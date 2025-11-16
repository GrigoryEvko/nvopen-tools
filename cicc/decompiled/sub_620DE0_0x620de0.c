// Function: sub_620DE0
// Address: 0x620de0
//
_WORD *__fastcall sub_620DE0(_WORD *a1, unsigned __int64 a2)
{
  _WORD *result; // rax
  _WORD *v3; // rdx

  result = a1 + 7;
  do
  {
    v3 = result;
    *result = a2;
    a2 >>= 16;
    --result;
  }
  while ( v3 != a1 );
  return result;
}
