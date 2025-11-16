// Function: sub_33CF8A0
// Address: 0x33cf8a0
//
__int64 __fastcall sub_33CF8A0(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 56);
  if ( !result )
    return 0;
  while ( a2 != *(_DWORD *)(result + 8) )
  {
    result = *(_QWORD *)(result + 32);
    if ( !result )
      return result;
  }
  return 1;
}
