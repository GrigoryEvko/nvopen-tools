// Function: sub_65C040
// Address: 0x65c040
//
__int64 __fastcall sub_65C040(__int64 a1)
{
  char v1; // al
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rax

  v1 = *(_BYTE *)(a1 + 268);
  *(_BYTE *)(a1 + 121) |= 0x80u;
  *(_BYTE *)(a1 + 269) = v1;
  v2 = *(_QWORD *)(a1 + 272);
  *(_QWORD *)(a1 + 280) = v2;
  *(_QWORD *)(a1 + 288) = v2;
  if ( (*(_BYTE *)(a1 + 123) & 0x10) != 0 )
  {
    *(_QWORD *)(a1 + 120) = *(_QWORD *)(a1 + 120) & 0xFFFFFF7FEFFFFFFFLL
                          | ((unsigned __int64)(*(_BYTE *)(a1 + 125) & 1) << 39);
    v6 = *(_QWORD *)(a1 + 304);
    *(_QWORD *)(a1 + 272) = v6;
    *(_QWORD *)(a1 + 280) = v6;
    *(_QWORD *)(a1 + 288) = v6;
    *(_QWORD *)(a1 + 56) = unk_4F077C8;
  }
  v3 = unk_4F077C8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 68) = 0;
  *(_QWORD *)(a1 + 40) = v3;
  *(_QWORD *)(a1 + 48) = v3;
  *(_QWORD *)(a1 + 120) &= 0xC3FFFFE71FFCFFFFLL;
  *(_QWORD *)(a1 + 128) &= 0xFFFF21FFEFEC04FFLL;
  *(_OWORD *)(a1 + 136) = 0;
  LODWORD(v3) = dword_4F077BC;
  *(_OWORD *)(a1 + 152) = 0;
  *(_OWORD *)(a1 + 168) = 0;
  if ( (_DWORD)v3 && qword_4F077A8 <= 0x9F5Fu )
    *(_BYTE *)(a1 + 178) |= 1u;
  v4 = unk_4F077C8;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = v4;
  result = sub_6E1BE0(a1 + 328);
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  return result;
}
