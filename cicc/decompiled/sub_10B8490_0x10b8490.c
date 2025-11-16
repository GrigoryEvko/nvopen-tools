// Function: sub_10B8490
// Address: 0x10b8490
//
_QWORD *__fastcall sub_10B8490(_QWORD *a1, __int64 *a2)
{
  _QWORD *result; // rax

  if ( *((_DWORD *)a1 + 2) > 0x40u )
    return sub_C43C10(a1, a2);
  result = (_QWORD *)*a2;
  *a1 ^= *a2;
  return result;
}
