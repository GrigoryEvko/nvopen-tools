// Function: sub_33C7E50
// Address: 0x33c7e50
//
__int64 __fastcall sub_33C7E50(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a1 > *a2;
  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  return result;
}
