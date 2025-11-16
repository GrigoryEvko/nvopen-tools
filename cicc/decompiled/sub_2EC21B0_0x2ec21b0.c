// Function: sub_2EC21B0
// Address: 0x2ec21b0
//
bool __fastcall sub_2EC21B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  char v7; // al

  v6 = *(_DWORD *)(a1 + 44);
  if ( (v6 & 4) == 0 && (v6 & 8) != 0 )
    v7 = sub_2E88A90(a1, 128, 1);
  else
    v7 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 7;
  return v7
      || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a4 + 1016LL))(
           a4,
           a1,
           a2,
           a3)
      || *(_WORD *)(a1 + 68) == 43;
}
