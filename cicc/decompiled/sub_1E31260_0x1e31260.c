// Function: sub_1E31260
// Address: 0x1e31260
//
__int64 __fastcall sub_1E31260(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // r13

  result = (*(_BYTE *)(a1 + 3) & 0x10) != 0;
  if ( (_BYTE)result != a2 )
  {
    result = *(_QWORD *)(a1 + 16);
    if ( result && (result = *(_QWORD *)(result + 24)) != 0 && (result = *(_QWORD *)(result + 56)) != 0 )
    {
      v3 = *(_QWORD *)(result + 40);
      sub_1E69A50(v3, a1);
      *(_BYTE *)(a1 + 3) = *(_BYTE *)(a1 + 3) & 0xEF | (16 * (a2 & 1));
      return sub_1E699D0(v3, a1);
    }
    else
    {
      *(_BYTE *)(a1 + 3) = *(_BYTE *)(a1 + 3) & 0xEF | (16 * (a2 & 1));
    }
  }
  return result;
}
