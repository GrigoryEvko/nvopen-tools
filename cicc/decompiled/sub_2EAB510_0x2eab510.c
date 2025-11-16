// Function: sub_2EAB510
// Address: 0x2eab510
//
void __fastcall sub_2EAB510(unsigned int *a1, unsigned int a2, unsigned int a3, int a4)
{
  int v5; // ebx
  unsigned int v6; // ecx

  v5 = (a4 << 8) & 0xFFF00;
  sub_2EAB370((__int64)a1);
  v6 = *a1;
  a1[6] = a2;
  a1[7] = a3;
  *a1 = v6 & 0xFFF00000 | v5 | 0x14;
}
