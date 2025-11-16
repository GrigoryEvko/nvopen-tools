// Function: sub_2E56370
// Address: 0x2e56370
//
__int64 **__fastcall sub_2E56370(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx

  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  *(_QWORD *)(a1 + 16) = 0x100000008LL;
  *(_BYTE *)(a1 + 28) = 1;
  v6 = *(_QWORD *)(a2 + 112);
  v7 = *(unsigned int *)(a2 + 120);
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)a1 = 1;
  v8 = v6 + 8 * v7;
  *(_QWORD *)(a1 + 120) = v6;
  *(_QWORD *)(a1 + 112) = v8;
  *(_QWORD *)(a1 + 128) = a2;
  *(_DWORD *)(a1 + 104) = 1;
  return sub_2DACB60(a1, a2, v8, a4, a5, a6);
}
