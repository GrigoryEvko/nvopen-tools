// Function: sub_167F890
// Address: 0x167f890
//
__int16 *__fastcall sub_167F890(__int16 *a1)
{
  __int16 v1; // ax

  v1 = *a1;
  *((_DWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  v1 &= 0xC000u;
  *((_BYTE *)a1 + 24) = 0;
  HIBYTE(v1) |= 0x20u;
  *((_QWORD *)a1 + 6) = 0;
  *a1 = v1;
  *((_QWORD *)a1 + 1) = a1 + 12;
  *((_QWORD *)a1 + 5) = a1 + 28;
  *((_BYTE *)a1 + 56) = 0;
  *((_QWORD *)a1 + 9) = 0;
  *((_QWORD *)a1 + 10) = 0;
  *((_QWORD *)a1 + 11) = 0;
  return a1 + 28;
}
