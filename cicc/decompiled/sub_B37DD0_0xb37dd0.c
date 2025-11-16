// Function: sub_B37DD0
// Address: 0xb37dd0
//
__int64 __fastcall sub_B37DD0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax

  v8 = sub_AE4420(a2, *(_QWORD *)(a1 + 72), *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8);
  v9 = sub_AD64C0(v8, a4, 0);
  return sub_B37B00(a1, a2, a3, v9, a5);
}
