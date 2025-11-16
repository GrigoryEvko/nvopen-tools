// Function: sub_284F380
// Address: 0x284f380
//
__int64 __fastcall sub_284F380(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
