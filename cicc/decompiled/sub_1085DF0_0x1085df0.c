// Function: sub_1085DF0
// Address: 0x1085df0
//
char __fastcall sub_1085DF0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rdi
  char v7; // dl
  __int64 v8; // rax
  int v9; // eax
  bool v10; // dl

  v6 = a1 + 96;
  *(_QWORD *)(v6 - 96) = a2;
  *(_QWORD *)(v6 - 88) = a3;
  *(_DWORD *)(v6 - 80) = 1;
  *(_QWORD *)(v6 - 56) = 0;
  *(_QWORD *)(v6 - 48) = 0;
  *(_QWORD *)(v6 - 40) = 0;
  *(_QWORD *)(v6 - 32) = 0;
  *(_QWORD *)(v6 - 24) = 0;
  *(_QWORD *)(v6 - 16) = 0;
  *(_QWORD *)(v6 - 8) = 0;
  *(_OWORD *)(v6 - 72) = 0;
  sub_C0BFB0(v6, 1, 0);
  *(_BYTE *)(a1 + 241) = 0;
  v7 = 1;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  v8 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 244) = a4;
  v9 = *(_DWORD *)(v8 + 8);
  *(_WORD *)(a1 + 24) = v9;
  if ( (_WORD)v9 != 0xAA64 )
  {
    v10 = (_WORD)v9 == 0xA641;
    LOBYTE(v9) = (_WORD)v9 == 0xA64E;
    v7 = v9 | v10;
  }
  *(_BYTE *)(a1 + 241) = v7;
  return v9;
}
