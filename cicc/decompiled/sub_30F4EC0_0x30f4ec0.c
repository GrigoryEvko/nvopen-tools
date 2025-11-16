// Function: sub_30F4EC0
// Address: 0x30f4ec0
//
char __fastcall sub_30F4EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char result; // al

  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = 0x300000000LL;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0x300000000LL;
  *(_QWORD *)(a1 + 104) = a4;
  result = sub_30F48E0(a1, a3);
  *(_BYTE *)a1 = result;
  return result;
}
