// Function: sub_143D870
// Address: 0x143d870
//
__int64 __fastcall sub_143D870(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4F99CC4;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_49EB8D8;
  *(_QWORD *)(a1 + 160) = 0;
  v2 = sub_163A1D0(a1, a2);
  return sub_143D780(v2);
}
