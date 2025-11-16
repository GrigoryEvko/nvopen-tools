// Function: sub_3243680
// Address: 0x3243680
//
__int64 __fastcall sub_3243680(__int64 a1)
{
  void (__fastcall *v2)(__int64, _QWORD, _QWORD); // r12
  unsigned __int8 v3; // al
  unsigned int v4; // eax
  int v5; // edx

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
  v2 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 8LL);
  v3 = sub_3736270(*(_QWORD *)(a1 + 16), 163);
  v2(a1, v3, 0);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v4);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL))(a1);
  v5 = *(unsigned __int16 *)(a1 + 100);
  *(_BYTE *)(a1 + 8) = 0;
  v5 &= ~0x40u;
  *(_WORD *)(a1 + 100) = v5;
  *(_BYTE *)(a1 + 100) = ((unsigned __int8)v5 >> 3) & 7 | v5 & 0xF8;
  return ((unsigned __int8)v5 >> 3) & 7 | v5 & 0xFFFFFFF8;
}
