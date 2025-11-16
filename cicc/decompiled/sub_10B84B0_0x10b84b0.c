// Function: sub_10B84B0
// Address: 0x10b84b0
//
_QWORD *__fastcall sub_10B84B0(_QWORD *a1, __int64 *a2)
{
  _QWORD *result; // rax

  if ( *((_DWORD *)a1 + 2) > 0x40u )
    return sub_C43BD0(a1, a2);
  result = (_QWORD *)*a2;
  *a1 |= *a2;
  return result;
}
