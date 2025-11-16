// Function: sub_1423A60
// Address: 0x1423a60
//
__int64 __fastcall sub_1423A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  sub_1423A40((_QWORD *)a1, a2);
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)a1 = &unk_49EB390;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x2000000000LL;
  *(_QWORD *)(a1 + 2112) = 0;
  *(_QWORD *)(a1 + 2120) = 0;
  *(_QWORD *)(a1 + 2128) = 0;
  *(_DWORD *)(a1 + 2136) = 0;
  return 0x2000000000LL;
}
