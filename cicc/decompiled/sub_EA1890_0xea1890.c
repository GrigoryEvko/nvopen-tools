// Function: sub_EA1890
// Address: 0xea1890
//
int *__fastcall sub_EA1890(int *a1)
{
  int v1; // eax
  int v2; // eax

  v1 = *a1;
  *((_BYTE *)a1 + 248) &= 0xFCu;
  *((_BYTE *)a1 + 12) = 0;
  v2 = v1 & 0xC000;
  *((_QWORD *)a1 + 2) = 2;
  BYTE1(v2) |= 0x10u;
  *((_QWORD *)a1 + 3) = 2;
  *a1 = v2;
  *((_WORD *)a1 + 2) = 1;
  *((_QWORD *)a1 + 4) = a1 + 12;
  *((_QWORD *)a1 + 8) = a1 + 20;
  *((_QWORD *)a1 + 12) = a1 + 28;
  *((_QWORD *)a1 + 16) = a1 + 36;
  *((_QWORD *)a1 + 20) = a1 + 44;
  *((_QWORD *)a1 + 5) = 0;
  *((_BYTE *)a1 + 48) = 0;
  *((_QWORD *)a1 + 9) = 0;
  *((_BYTE *)a1 + 80) = 0;
  *((_QWORD *)a1 + 13) = 0;
  *((_BYTE *)a1 + 112) = 0;
  *((_QWORD *)a1 + 17) = 0;
  *((_BYTE *)a1 + 144) = 0;
  *((_QWORD *)a1 + 21) = 0;
  *((_BYTE *)a1 + 176) = 0;
  *((_QWORD *)a1 + 24) = a1 + 52;
  *((_QWORD *)a1 + 25) = 0;
  *((_BYTE *)a1 + 208) = 0;
  *((_QWORD *)a1 + 28) = 0;
  *((_QWORD *)a1 + 29) = 0;
  *((_QWORD *)a1 + 30) = 0;
  return a1 + 52;
}
