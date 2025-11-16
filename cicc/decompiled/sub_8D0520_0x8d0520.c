// Function: sub_8D0520
// Address: 0x8d0520
//
__int64 __fastcall sub_8D0520(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = 1;
  if ( a1 != a2 )
  {
    if ( a1 && a2 )
    {
      result = dword_4F07588;
      if ( dword_4F07588 )
        return (*(_QWORD *)(a2 + 32) == *(_QWORD *)(a1 + 32)) & (unsigned __int8)(*(_QWORD *)(a1 + 32) != 0);
    }
    else
    {
      return 0;
    }
  }
  return result;
}
