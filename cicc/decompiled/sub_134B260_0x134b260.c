// Function: sub_134B260
// Address: 0x134b260
//
__int64 __fastcall sub_134B260(_QWORD *a1)
{
  __int64 result; // rax

  result = a1[103] - (a1[105] + a1[29]);
  if ( result > a1[28] )
    a1[28] = result;
  return result;
}
