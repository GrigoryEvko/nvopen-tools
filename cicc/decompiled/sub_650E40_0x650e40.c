// Function: sub_650E40
// Address: 0x650e40
//
__int64 __fastcall sub_650E40(__int64 a1)
{
  __int64 result; // rax
  _QWORD **v2; // r12

  result = (unsigned int)dword_4D043E0;
  if ( dword_4D043E0 )
  {
    result = (__int64)word_4F06418;
    if ( word_4F06418[0] == 142 )
    {
      v2 = (_QWORD **)(a1 + 200);
      if ( *(_QWORD *)(a1 + 200) )
        v2 = sub_5CB9F0((_QWORD **)(a1 + 200));
      result = sub_5CC970(10);
      *v2 = (_QWORD *)result;
    }
  }
  return result;
}
