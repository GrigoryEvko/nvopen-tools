// Function: sub_2FF1980
// Address: 0x2ff1980
//
__int64 __fastcall sub_2FF1980(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rax

  result = sub_2FF0570(a1);
  if ( (_DWORD)result )
  {
    if ( !byte_5029108 )
    {
      v2 = (_QWORD *)sub_2D5CDB0();
      return sub_2FF0E80(a1, v2, 0);
    }
  }
  return result;
}
