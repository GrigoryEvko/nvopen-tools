// Function: sub_1D18C40
// Address: 0x1d18c40
//
__int64 __fastcall sub_1D18C40(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 48);
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
