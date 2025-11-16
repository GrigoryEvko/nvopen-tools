// Function: sub_2CBEE80
// Address: 0x2cbee80
//
__int64 __fastcall sub_2CBEE80(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // rcx
  __int64 result; // rax
  __int64 v7; // rcx

  v5 = *(unsigned __int16 *)(a2 + 2);
  if ( (((*(_WORD *)(a2 + 2) & 0x3F) - 37) & 0xFFFB) != 0 )
  {
    v7 = v5 & 0x3B;
    if ( (_WORD)v7 == 35 && a3 == *(_QWORD *)(a2 - 32) )
    {
      result = sub_D48480(a1, *(_QWORD *)(a2 - 64), a3, v7);
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
  result = sub_D48480(a1, *(_QWORD *)(a2 - 32), a3, v5);
  if ( !(_BYTE)result )
    return 0;
  *a4 = *(_QWORD *)(a2 - 32);
  return result;
}
