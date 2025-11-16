// Function: sub_33CCCC0
// Address: 0x33cccc0
//
_QWORD *__fastcall sub_33CCCC0(__int64 a1, __int64 a2, __int64 *a3)
{
  _QWORD *result; // rax

  result = sub_C65B40(a1 + 520, a2, a3, (__int64)off_4A367D0);
  if ( result )
  {
    if ( (unsigned int)(*((_DWORD *)result + 6) - 11) <= 1 )
      BUG();
  }
  return result;
}
