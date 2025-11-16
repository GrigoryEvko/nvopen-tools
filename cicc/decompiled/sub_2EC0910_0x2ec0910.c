// Function: sub_2EC0910
// Address: 0x2ec0910
//
__int64 __fastcall sub_2EC0910(__int64 a1, __int64 a2, _BYTE *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  v6 = sub_2EC0130(a1, a3, a4, a4, a5, a6);
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (v6 & 0x7FFFFFFF)) = 4;
  sub_2EBE740(a1, v6, a2, v7, v8, v9);
  sub_2EBDFC0(a1, v6);
  return v6;
}
