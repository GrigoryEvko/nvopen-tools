// Function: sub_5EA7A0
// Address: 0x5ea7a0
//
__int64 __fastcall sub_5EA7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6, __int64 a7)
{
  char v8; // bl
  __int64 v9; // rax
  char v10; // r8

  v8 = (8 * (a6 & 1)) | (4 * (a5 & 1) + 2);
  v9 = sub_5E4B20(a4);
  v10 = *(_BYTE *)(v9 + 184);
  *(_QWORD *)(v9 + 16) = a1;
  *(_QWORD *)(v9 + 136) = a2;
  *(_QWORD *)(v9 + 144) = a3;
  *(_QWORD *)(v9 + 128) = a7;
  *(_BYTE *)(v9 + 184) = v10 & 0xF1 | v8;
  return sub_5E9580(v9);
}
