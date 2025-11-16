// Function: sub_1869D00
// Address: 0x1869d00
//
__int64 __fastcall sub_1869D00(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
