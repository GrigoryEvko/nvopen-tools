// Function: sub_2EAB3B0
// Address: 0x2eab3b0
//
void __fastcall sub_2EAB3B0(__int64 a1, __int64 a2, int a3)
{
  int v3; // ebx
  int v4; // edx

  v3 = (a3 << 8) & 0xFFF00 | 1;
  sub_2EAB370(a1);
  v4 = *(_DWORD *)a1;
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)a1 = v4 & 0xFFF00000 | v3;
}
