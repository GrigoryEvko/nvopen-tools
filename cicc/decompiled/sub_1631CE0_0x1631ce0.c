// Function: sub_1631CE0
// Address: 0x1631ce0
//
__int64 __fastcall sub_1631CE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = a1 - 56;
  *(_QWORD *)(a2 + 40) = a1 - 56;
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    if ( v3 )
      return sub_164D6D0(v3, a2);
  }
  return result;
}
