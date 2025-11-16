// Function: sub_2EAB460
// Address: 0x2eab460
//
void __fastcall sub_2EAB460(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  int v5; // ebx
  int v6; // ecx

  v5 = (a4 << 8) & 0xFFF00;
  sub_2EAB370(a1);
  v6 = *(_DWORD *)a1;
  *(_DWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 32) = HIDWORD(a3);
  *(_DWORD *)a1 = v6 & 0xFFF00000 | v5 | 0xA;
}
