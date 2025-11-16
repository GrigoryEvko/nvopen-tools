// Function: sub_CFB860
// Address: 0xcfb860
//
__int64 __fastcall sub_CFB860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax

  v5 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 8) = v5 + 8;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_BYTE *)(a1 + 192) = 0;
  return a1;
}
