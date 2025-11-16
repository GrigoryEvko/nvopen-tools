// Function: sub_2FAF2F0
// Address: 0x2faf2f0
//
__int64 __fastcall sub_2FAF2F0(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax

  result = (a2 >> 13) + ((a2 >> 12) & 1);
  if ( !result )
    result = 1;
  *(_QWORD *)(a1 + 216) = result;
  return result;
}
