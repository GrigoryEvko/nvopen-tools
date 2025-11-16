// Function: sub_2255E10
// Address: 0x2255e10
//
__int64 __fastcall sub_2255E10(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax

  result = __strftime_l();
  if ( !result )
    *a2 = 0;
  return result;
}
