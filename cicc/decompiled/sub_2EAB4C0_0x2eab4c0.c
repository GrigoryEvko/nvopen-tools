// Function: sub_2EAB4C0
// Address: 0x2eab4c0
//
void __fastcall sub_2EAB4C0(unsigned int *a1, unsigned int a2, int a3)
{
  int v3; // ebx
  unsigned int v4; // edx

  v3 = (a3 << 8) & 0xFFF00 | 5;
  sub_2EAB370((__int64)a1);
  v4 = *a1;
  a1[6] = a2;
  *a1 = v4 & 0xFFF00000 | v3;
}
