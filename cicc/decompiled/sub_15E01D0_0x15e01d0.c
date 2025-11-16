// Function: sub_15E01D0
// Address: 0x15e01d0
//
__int64 __fastcall sub_15E01D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = sub_157F970(a2, a1 - 72);
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    if ( v3 )
      return sub_164D6D0(v3, a2);
  }
  return result;
}
