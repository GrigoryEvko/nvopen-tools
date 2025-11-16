// Function: sub_621DB0
// Address: 0x621db0
//
_WORD *__fastcall sub_621DB0(_WORD *a1)
{
  _WORD *result; // rax
  _WORD *v2; // rdx

  result = a1 + 7;
  do
  {
    v2 = result;
    *result = ~*result;
    --result;
  }
  while ( v2 != a1 );
  return result;
}
