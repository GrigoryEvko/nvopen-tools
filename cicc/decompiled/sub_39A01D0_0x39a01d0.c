// Function: sub_39A01D0
// Address: 0x39a01d0
//
_QWORD *__fastcall sub_39A01D0(__int64 *a1, _QWORD *a2, unsigned __int8 a3)
{
  _QWORD *result; // rax

  result = (_QWORD *)a2[10];
  if ( *((_DWORD *)result + 9) != 3 )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*a1 + 256) + 160LL))(
      *(_QWORD *)(*a1 + 256),
      a2[7],
      0);
    (*(void (__fastcall **)(_QWORD *, _QWORD))(*a2 + 32LL))(a2, a3);
    return sub_397C610(*a1, (__int64)(a2 + 1));
  }
  return result;
}
