// Function: sub_2102D20
// Address: 0x2102d20
//
__int64 __fastcall sub_2102D20(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_4FCF954;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
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
  *(_DWORD *)(a1 + 256) = 0;
  *(_QWORD *)a1 = &unk_4A00C40;
  *(_QWORD *)(a1 + 288) = a1 + 304;
  *(_QWORD *)(a1 + 296) = 0x400000000LL;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 1;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 424) = 0;
  return a1 + 352;
}
