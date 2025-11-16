// Function: sub_140CFB0
// Address: 0x140cfb0
//
__int64 __fastcall sub_140CFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = a4;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 88) = a2;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = a1 + 184;
  *(_QWORD *)(a1 + 160) = a1 + 184;
  *(_QWORD *)(a1 + 168) = 8;
  *(_DWORD *)(a1 + 176) = 0;
  *(_BYTE *)(a1 + 248) = a5;
  return a1 + 184;
}
