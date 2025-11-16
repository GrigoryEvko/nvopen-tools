// Function: sub_39A0020
// Address: 0x39a0020
//
__int64 __fastcall sub_39A0020(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 result; // rax

  a1[3] = (__int64)(a1 + 5);
  a1[4] = 0x400000000LL;
  a1[9] = (__int64)(a1 + 11);
  v9 = a1 + 1;
  v10 = (__int64)(a1 + 15);
  *(_QWORD *)(v10 - 120) = a2;
  *(_QWORD *)(v10 - 8) = v9;
  *(_QWORD *)(v10 - 112) = 0;
  *(_QWORD *)(v10 - 104) = 0;
  *(_QWORD *)(v10 - 40) = 0;
  *(_QWORD *)(v10 - 32) = 0;
  *(_QWORD *)(v10 - 24) = 1;
  sub_16BD940(v10, 6);
  v11 = *a1;
  a1[18] = 0;
  a1[19] = 0;
  a1[20] = 0;
  a1[15] = (__int64)&unk_4A3FC58;
  a1[21] = (__int64)(a1 + 23);
  a1[22] = 0x100000000LL;
  result = sub_39A11A0(a1 + 24, a5, v11, a3, a4);
  a1[31] = 0;
  a1[32] = 0;
  a1[33] = 0;
  a1[34] = 0;
  a1[35] = 0;
  *((_DWORD *)a1 + 72) = 0;
  a1[37] = 0;
  a1[38] = 0;
  a1[39] = 0;
  *((_DWORD *)a1 + 80) = 0;
  a1[41] = 0;
  a1[42] = 0;
  a1[43] = 0;
  *((_DWORD *)a1 + 88) = 0;
  a1[45] = 0;
  a1[46] = 0;
  a1[47] = 0;
  *((_DWORD *)a1 + 96) = 0;
  return result;
}
