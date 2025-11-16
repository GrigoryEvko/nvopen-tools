// Function: sub_2EAB250
// Address: 0x2eab250
//
__int64 __fastcall sub_2EAB250(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // r13

  result = (*(_BYTE *)(a1 + 3) & 0x10) != 0;
  if ( (_BYTE)result != a2 )
  {
    result = *(_QWORD *)(a1 + 16);
    if ( result && (result = *(_QWORD *)(result + 24)) != 0 && (result = *(_QWORD *)(result + 32)) != 0 )
    {
      v3 = *(_QWORD *)(result + 32);
      sub_2EBEB60(v3, a1);
      *(_BYTE *)(a1 + 3) = *(_BYTE *)(a1 + 3) & 0xEF | (16 * (a2 & 1));
      return sub_2EBEAE0(v3, a1);
    }
    else
    {
      *(_BYTE *)(a1 + 3) = *(_BYTE *)(a1 + 3) & 0xEF | (16 * (a2 & 1));
    }
  }
  return result;
}
