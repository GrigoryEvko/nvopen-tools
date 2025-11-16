// Function: sub_7AB700
// Address: 0x7ab700
//
__int64 __fastcall sub_7AB700(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = 0xFFFFFFFFLL;
  if ( *a2 <= *a1 )
    return a2[1] < *a1;
  return result;
}
