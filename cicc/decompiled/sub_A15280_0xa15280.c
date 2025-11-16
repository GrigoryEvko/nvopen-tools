// Function: sub_A15280
// Address: 0xa15280
//
__int64 __fastcall sub_A15280(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
