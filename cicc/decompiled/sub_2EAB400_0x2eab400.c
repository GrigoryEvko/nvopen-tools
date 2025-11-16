// Function: sub_2EAB400
// Address: 0x2eab400
//
void __fastcall sub_2EAB400(__int64 a1, __int64 a2, int a3)
{
  int v3; // ebx
  int v4; // edx

  v3 = (a3 << 8) & 0xFFF00 | 9;
  sub_2EAB370(a1);
  v4 = *(_DWORD *)a1;
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)a1 = v4 & 0xFFF00000 | v3;
}
