// Function: sub_325F530
// Address: 0x325f530
//
_QWORD *__fastcall sub_325F530(_QWORD *a1, __int64 *a2)
{
  _QWORD *result; // rax

  if ( *((_DWORD *)a1 + 2) > 0x40u )
    return sub_C43B90(a1, a2);
  result = (_QWORD *)*a2;
  *a1 &= *a2;
  return result;
}
