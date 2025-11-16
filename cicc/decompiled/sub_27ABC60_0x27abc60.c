// Function: sub_27ABC60
// Address: 0x27abc60
//
__int64 __fastcall sub_27ABC60(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
