// Function: sub_1368BD0
// Address: 0x1368bd0
//
__int64 __fastcall sub_1368BD0(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    return *(_QWORD *)(result + 128);
  return result;
}
