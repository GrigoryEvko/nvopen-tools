// Function: sub_F8E3B0
// Address: 0xf8e3b0
//
__int64 __fastcall sub_F8E3B0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a1 > *a2;
  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  return result;
}
