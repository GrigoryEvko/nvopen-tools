// Function: sub_24C0A60
// Address: 0x24c0a60
//
void __fastcall sub_24C0A60(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)(a1 + 8) = a3;
  *(_WORD *)a1 = a2;
  *(_BYTE *)(a1 + 2) = BYTE2(a2);
  *(_QWORD *)(a1 + 16) = a4;
}
