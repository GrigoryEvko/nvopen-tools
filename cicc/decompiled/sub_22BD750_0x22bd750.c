// Function: sub_22BD750
// Address: 0x22bd750
//
__int64 __fastcall sub_22BD750(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // r13
  char v6; // bl
  char v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  char v12; // [rsp+Fh] [rbp-31h]

  v12 = byte_4FDBAC8;
  v5 = qword_4FDBB68[8];
  v6 = unk_4FDB9E8 ^ 1;
  v7 = qword_4FDBC48[8];
  v8 = a1 + 32;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  v9 = a1 + 96;
  *(_QWORD *)(v9 - 80) = v8;
  *(_QWORD *)(v9 - 72) = 0x400000000LL;
  *(_QWORD *)(v9 - 96) = 0;
  *(_QWORD *)(v9 - 88) = 0;
  *(_QWORD *)(a1 + 120) = 0x400000000LL;
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_DWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 280) = v9;
  *(_QWORD *)(a1 + 272) = a1;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 16;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 4294967293LL;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 0;
  v10 = sub_9D1E70(v9, 16, 16, 3);
  *(_QWORD *)(v10 + 8) = v10;
  *(_QWORD *)v10 = v10 | 4;
  *(_BYTE *)(a1 + 304) = v7 ^ 1;
  *(_BYTE *)(a1 + 305) = v5 ^ 1;
  *(_BYTE *)(a1 + 307) = v6;
  *(_QWORD *)(a1 + 288) = v10;
  *(_BYTE *)(a1 + 306) = v12;
  *(_BYTE *)(a1 + 308) = 0;
  *(_BYTE *)(a1 + 336) = 0;
  sub_22BD540(a1, a3);
  return a1;
}
