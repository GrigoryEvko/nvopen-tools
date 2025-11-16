// Function: sub_8808B0
// Address: 0x8808b0
//
_BOOL8 __fastcall sub_8808B0(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // rsi

  result = 1;
  if ( *(_BYTE *)(a2 + 80) == 24 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    v4 = *(_QWORD *)(a2 + 88);
    if ( v3 )
      v5 = *(_QWORD *)(v3 + 128);
    else
      v5 = *(_QWORD *)(qword_4F04C68[0] + 184LL);
    result = 1;
    if ( *(_DWORD *)(v4 + 40) != *(_DWORD *)(a1 + 40) )
      return (unsigned int)sub_880800(v4, v5) != 0;
  }
  return result;
}
