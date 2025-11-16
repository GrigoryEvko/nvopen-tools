// Function: sub_BA86C0
// Address: 0xba86c0
//
__int64 __fastcall sub_BA86C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = a1 - 56;
  *(_QWORD *)(a2 + 40) = a1 - 56;
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    if ( v3 )
      return sub_BD8920(v3, a2);
  }
  return result;
}
