// Function: sub_20F7A40
// Address: 0x20f7a40
//
unsigned __int64 *__fastcall sub_20F7A40(
        unsigned __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 *result; // rax

  a1[1] = a3;
  a1[2] = a2;
  *a1 = a6;
  sub_20F79C0(a1);
  result = a1 + 6;
  do
  {
    *(_DWORD *)result = 0;
    result += 90;
    *(result - 88) = a2;
    *(result - 87) = a4;
    *(result - 86) = a5;
  }
  while ( result != a1 + 2886 );
  return result;
}
