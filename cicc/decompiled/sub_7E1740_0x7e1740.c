// Function: sub_7E1740
// Address: 0x7e1740
//
void __fastcall sub_7E1740(__int64 a1, __int64 a2)
{
  *(_DWORD *)a2 = 1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_BYTE *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 8) = a1;
}
