// Function: sub_16113E0
// Address: 0x16113e0
//
void *__fastcall sub_16113E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 6;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EDE80;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4F9E3BA;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 160) = &unk_49EDA68;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = 0x1000000000LL;
  *(_QWORD *)(a1 + 424) = 0x1000000000LL;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  v1 = sub_22077B0(568);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_DWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 16) = &unk_4F9E389;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_QWORD *)(v1 + 184) = v1 + 200;
    v2 = v1 + 160;
    *(_QWORD *)(v1 + 416) = v1 + 432;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 192) = 0x1000000000LL;
    *(_QWORD *)(v1 + 424) = 0x1000000000LL;
    *(_DWORD *)(v1 + 560) = 0;
    *(_QWORD *)(v1 + 328) = 0;
    *(_QWORD *)(v1 + 336) = 0;
    *(_QWORD *)(v1 + 344) = 0;
    *(_QWORD *)(v1 + 352) = 0;
    *(_QWORD *)(v1 + 360) = 0;
    *(_QWORD *)(v1 + 368) = 0;
    *(_QWORD *)(v1 + 376) = 0;
    *(_QWORD *)(v1 + 384) = 1;
    *(_QWORD *)(v1 + 392) = 0;
    *(_QWORD *)(v1 + 400) = 0;
    *(_DWORD *)(v1 + 408) = 0;
    *(_QWORD *)v1 = &unk_49EDC10;
    *(_QWORD *)(v1 + 160) = &unk_49EDCC8;
  }
  sub_1611190(a1 + 568, v2);
  *(_BYTE *)(a1 + 1304) = 0;
  *(_QWORD *)a1 = &unk_49ED7E8;
  *(_QWORD *)(a1 + 160) = &unk_49ED8A0;
  *(_QWORD *)(a1 + 568) = &unk_49ED8E0;
  return &unk_49ED8E0;
}
