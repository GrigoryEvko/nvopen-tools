// Function: sub_15E0220
// Address: 0x15e0220
//
__int64 __fastcall sub_15E0220(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  result = sub_157F970(a2, 0);
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    if ( v3 )
    {
      v4 = sub_16498B0(a2);
      return sub_164D860(v3, v4);
    }
  }
  return result;
}
