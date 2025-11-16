// Function: sub_7E1E20
// Address: 0x7e1e20
//
__int64 __fastcall sub_7E1E20(__int64 a1)
{
  __int64 result; // rax

  while ( *(_BYTE *)(a1 + 140) == 12 )
  {
    result = *(_QWORD *)(a1 + 176);
    if ( result )
      return result;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return a1;
}
