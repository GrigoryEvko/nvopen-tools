// Function: sub_2EC0720
// Address: 0x2ec0720
//
__int64 __fastcall sub_2EC0720(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, unsigned __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  v7 = sub_2EC0130(a1, a4, a5, (__int64)a4, a5, a6);
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (v7 & 0x7FFFFFFF)) = a2;
  sub_2EBE740(a1, v7, a3, v8, v9, v10);
  sub_2EBDFC0(a1, v7);
  return v7;
}
