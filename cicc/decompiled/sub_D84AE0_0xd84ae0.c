// Function: sub_D84AE0
// Address: 0xd84ae0
//
__int64 __fastcall sub_D84AE0(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 49) = 0;
  *(_BYTE *)(a1 + 51) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  sub_D84780((__int64 *)a1);
  return a1;
}
