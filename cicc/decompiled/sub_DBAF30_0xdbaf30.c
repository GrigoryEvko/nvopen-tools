// Function: sub_DBAF30
// Address: 0xdbaf30
//
_QWORD *__fastcall sub_DBAF30(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  result = sub_DB9E00(a1, a2);
  if ( !*((_BYTE *)result + 136) )
    return (_QWORD *)sub_DBA850(a1, a2);
  return result;
}
