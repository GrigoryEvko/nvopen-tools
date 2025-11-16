// Function: sub_2CBED80
// Address: 0x2cbed80
//
__int64 __fastcall sub_2CBED80(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int16 v5; // ax
  __int64 v6; // rcx
  __int64 result; // rax

  v5 = *(_WORD *)(a2 + 2);
  v6 = v5 & 0x3B;
  if ( (v5 & 0x3B) != 0x22 )
  {
    if ( (((v5 & 0x3F) - 36) & 0xFFFB) == 0 && a3 == *(_QWORD *)(a2 - 32) )
    {
      result = sub_D48480(a1, *(_QWORD *)(a2 - 64), a3, v6);
      if ( (_BYTE)result )
      {
        *a4 = *(_QWORD *)(a2 - 64);
        return result;
      }
    }
    return 0;
  }
  if ( a3 != *(_QWORD *)(a2 - 64) )
    return 0;
  result = sub_D48480(a1, *(_QWORD *)(a2 - 32), a3, v6);
  if ( !(_BYTE)result )
    return 0;
  *a4 = *(_QWORD *)(a2 - 32);
  return result;
}
