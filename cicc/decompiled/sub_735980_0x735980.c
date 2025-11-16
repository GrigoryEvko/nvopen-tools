// Function: sub_735980
// Address: 0x735980
//
__int64 __fastcall sub_735980(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rbx

  result = *(unsigned __int8 *)(a2 + 173);
  if ( (_BYTE)result == 10 )
  {
LABEL_7:
    for ( i = *(_QWORD *)(a2 + 176); i; i = *(_QWORD *)(i + 120) )
      result = sub_735980(a1, i);
  }
  else
  {
    while ( (_BYTE)result != 9 )
    {
      if ( (_BYTE)result != 11 )
        return result;
      a2 = *(_QWORD *)(a2 + 176);
      result = *(unsigned __int8 *)(a2 + 173);
      if ( (_BYTE)result == 10 )
        goto LABEL_7;
    }
    return sub_735890(a1, *(_QWORD *)(a2 + 176), 0);
  }
  return result;
}
