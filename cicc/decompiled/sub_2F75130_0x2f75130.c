// Function: sub_2F75130
// Address: 0x2f75130
//
__int64 __fastcall sub_2F75130(__int64 *a1)
{
  __int64 result; // rax

  a1[56] = 0;
  result = *a1;
  a1[55] = 0;
  if ( result != a1[1] )
    a1[1] = result;
  *((_DWORD *)a1 + 8) = 0;
  *((_DWORD *)a1 + 60) = 0;
  return result;
}
