// Function: sub_67C730
// Address: 0x67c730
//
__int64 __fastcall sub_67C730(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax

  if ( *a1 )
  {
    result = a1[1];
    *(_QWORD *)(result + 8) = a2;
  }
  else
  {
    *a1 = a2;
  }
  a1[1] = a2;
  return result;
}
