// Function: sub_1B42380
// Address: 0x1b42380
//
__int64 __fastcall sub_1B42380(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a1 > *a2;
  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  return result;
}
