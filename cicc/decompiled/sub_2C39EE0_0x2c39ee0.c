// Function: sub_2C39EE0
// Address: 0x2c39ee0
//
__int64 __fastcall sub_2C39EE0(__int64 *a1)
{
  char v1; // bl
  char v2; // bl
  __int64 result; // rax

  sub_2C32E80(a1);
  do
  {
    do
    {
      v1 = sub_2C394D0((__int64)a1);
      v2 = sub_2C32020(*a1) | v1;
      result = sub_2C34510(a1);
    }
    while ( v2 );
  }
  while ( (_BYTE)result );
  return result;
}
