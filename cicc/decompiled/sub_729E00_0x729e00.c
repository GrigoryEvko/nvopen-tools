// Function: sub_729E00
// Address: 0x729e00
//
_QWORD *__fastcall sub_729E00(unsigned int a1, _QWORD *a2, _QWORD *a3, _DWORD *a4, _DWORD *a5)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_729B10(a1, a4, a5, 0);
  if ( result )
  {
    *a2 = *result;
    *a3 = result[1];
  }
  else
  {
    *a3 = byte_3F871B3;
    *a2 = byte_3F871B3;
    *a4 = 0;
  }
  return result;
}
