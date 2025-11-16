// Function: sub_130D500
// Address: 0x130d500
//
_QWORD *__fastcall sub_130D500(_QWORD *a1)
{
  _QWORD *result; // rax

  result = a1;
  do
  {
    *result = 1;
    result += 2;
    *(result - 1) = 0;
  }
  while ( result != a1 + 32 );
  do
  {
    *result = 1;
    result += 2;
    *(result - 1) = 0;
  }
  while ( result != a1 + 48 );
  return result;
}
