// Function: sub_2C83D20
// Address: 0x2c83d20
//
bool __fastcall sub_2C83D20(__int64 a1)
{
  __int64 v1; // rax

  return *(_BYTE *)a1 == 85
      && (v1 = *(_QWORD *)(a1 - 32)) != 0
      && !*(_BYTE *)v1
      && *(_QWORD *)(v1 + 24) == *(_QWORD *)(a1 + 80)
      && (*(_BYTE *)(v1 + 33) & 0x20) != 0
      && sub_CEA1A0(*(_DWORD *)(v1 + 36));
}
