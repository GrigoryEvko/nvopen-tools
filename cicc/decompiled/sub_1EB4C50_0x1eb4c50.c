// Function: sub_1EB4C50
// Address: 0x1eb4c50
//
void *__fastcall sub_1EB4C50(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // rax

  *(_QWORD *)(a1 + 16) = &unk_4FC9150;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 8) = 0;
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
  *(_QWORD *)a1 = &unk_49FB790;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 8;
  v1 = (_QWORD *)malloc(8u);
  if ( !v1 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v1 = 0;
  }
  *(_QWORD *)(a1 + 160) = v1;
  *(_QWORD *)(a1 + 168) = 1;
  *v1 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 8;
  v2 = (_QWORD *)malloc(8u);
  if ( !v2 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v2 = 0;
  }
  *(_QWORD *)(a1 + 184) = v2;
  *(_QWORD *)(a1 + 192) = 1;
  *v2 = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v3 = (_QWORD *)malloc(8u);
  if ( !v3 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v3 = 0;
  }
  *(_QWORD *)(a1 + 208) = v3;
  *v3 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 232) = &unk_4A00E78;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  sub_1ED72C0(a1 + 280);
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = a1 + 416;
  *(_QWORD *)(a1 + 392) = a1 + 416;
  *(_QWORD *)a1 = off_49FD818;
  *(_QWORD *)(a1 + 232) = &unk_49FD910;
  *(_QWORD *)(a1 + 400) = 32;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 672) = &unk_49FD968;
  *(_QWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_DWORD *)(a1 + 744) = 0;
  return &unk_49FD968;
}
