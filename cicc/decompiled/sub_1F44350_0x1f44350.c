// Function: sub_1F44350
// Address: 0x1f44350
//
__int64 __fastcall sub_1F44350(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  char *v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // r9d
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v9[0] = sub_1560340((_QWORD *)(*(_QWORD *)a4 + 112LL), -1, "reciprocal-estimates", 0x14u);
  v5 = (char *)sub_155D8B0(v9);
  return sub_1F3DC60(1, a2, a3, v5, v6, v7);
}
