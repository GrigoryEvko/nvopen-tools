// Function: sub_D853E0
// Address: 0xd853e0
//
__int64 __fastcall sub_D853E0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
