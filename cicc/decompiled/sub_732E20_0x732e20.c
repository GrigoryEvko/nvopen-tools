// Function: sub_732E20
// Address: 0x732e20
//
__int64 __fastcall sub_732E20(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
  {
    v2 = 0;
  }
  else
  {
    result = dword_4F04C58;
    if ( dword_4F04C58 == -1 )
      return result;
    v2 = 776LL * dword_4F04C58;
  }
  result = qword_4F04C68[0] + v2;
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(result + 504);
  *(_QWORD *)(result + 504) = a1;
  return result;
}
