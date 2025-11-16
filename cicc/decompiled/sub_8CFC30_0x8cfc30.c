// Function: sub_8CFC30
// Address: 0x8cfc30
//
_QWORD *__fastcall sub_8CFC30(__int64 *a1)
{
  _QWORD *result; // rax

  result = &qword_4F074A0;
  if ( qword_4F074B0 == qword_4F60258 )
  {
    result = (_QWORD *)dword_4D03FC0;
    if ( dword_4D03FC0 )
    {
      if ( !a1[4] )
        return sub_8CC270(a1);
    }
  }
  return result;
}
