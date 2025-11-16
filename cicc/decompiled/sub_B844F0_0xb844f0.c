// Function: sub_B844F0
// Address: 0xb844f0
//
void *__fastcall sub_B844F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 5;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_QWORD *)a1 = &unk_49DAD88;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 16) = &unk_4F81902;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 176) = &unk_49DA9F0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 200) = 0x1000000000LL;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 424) = 0x1000000000LL;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 384) = 1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_OWORD *)(a1 + 336) = 0;
  *(_OWORD *)(a1 + 352) = 0;
  *(_OWORD *)(a1 + 368) = 0;
  v1 = sub_22077B0(568);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_DWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 16) = &unk_4F818FF;
    *(_QWORD *)(v1 + 56) = v1 + 104;
    *(_QWORD *)(v1 + 112) = v1 + 160;
    *(_QWORD *)(v1 + 192) = v1 + 208;
    v2 = v1 + 176;
    *(_QWORD *)(v1 + 416) = v1 + 432;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_DWORD *)(v1 + 144) = 1065353216;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 200) = 0x1000000000LL;
    *(_QWORD *)(v1 + 424) = 0x1000000000LL;
    *(_DWORD *)(v1 + 560) = 0;
    *(_QWORD *)(v1 + 384) = 1;
    *(_QWORD *)(v1 + 392) = 0;
    *(_QWORD *)(v1 + 400) = 0;
    *(_DWORD *)(v1 + 408) = 0;
    *(_OWORD *)(v1 + 336) = 0;
    *(_OWORD *)(v1 + 352) = 0;
    *(_QWORD *)v1 = &unk_49DAA78;
    *(_QWORD *)(v1 + 176) = &unk_49DAB30;
    *(_OWORD *)(v1 + 368) = 0;
  }
  sub_B842C0(a1 + 568, v2);
  *(_BYTE *)(a1 + 1288) = 0;
  *(_QWORD *)a1 = &unk_49DA770;
  *(_QWORD *)(a1 + 176) = &unk_49DA828;
  *(_QWORD *)(a1 + 568) = &unk_49DA868;
  return &unk_49DA868;
}
