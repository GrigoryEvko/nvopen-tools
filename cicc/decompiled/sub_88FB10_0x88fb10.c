// Function: sub_88FB10
// Address: 0x88fb10
//
_BOOL8 __fastcall sub_88FB10(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rdi
  __int64 v4; // rsi

  result = 1;
  if ( a1 != a2 )
  {
    if ( !a1 || !a2 )
      return 0;
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 104LL);
    v4 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 104LL);
    if ( v3 != v4 )
    {
      result = 0;
      if ( *qword_4D03FD0 )
      {
        if ( v3 && v4 )
          return (unsigned int)sub_8C7EB0(v3, v4) != 0;
        return 0;
      }
    }
  }
  return result;
}
