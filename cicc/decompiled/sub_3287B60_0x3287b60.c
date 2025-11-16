// Function: sub_3287B60
// Address: 0x3287b60
//
__int64 __fastcall sub_3287B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // rdx
  __int64 result; // rax

  v6 = a3;
  *(_QWORD *)(a1 + 48) = sub_33ECD10(1, a2, a3, a4, a5, a6);
  *(_QWORD *)(a1 + 64) = 0x100000000LL;
  *(_WORD *)(a1 + 34) = -1;
  v7 = a1 + 96;
  *(_WORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 328;
  *(_DWORD *)(a1 + 36) = -1;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 120) = 0;
  result = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 112) = a1;
  *(_QWORD *)(a1 + 96) = a2;
  *(_DWORD *)(a1 + 104) = v6;
  *(_QWORD *)(a1 + 128) = result;
  if ( result )
    *(_QWORD *)(result + 24) = a1 + 128;
  *(_QWORD *)(a2 + 56) = v7;
  *(_QWORD *)(a1 + 120) = a2 + 56;
  *(_DWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 40) = v7;
  return result;
}
