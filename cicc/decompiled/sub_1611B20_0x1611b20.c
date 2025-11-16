// Function: sub_1611B20
// Address: 0x1611b20
//
__int64 __fastcall sub_1611B20(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rsi
  _QWORD *v7; // rax
  __int64 v8; // rdi

  *a1 = &unk_49EDBC8;
  a1[2] = a2;
  v2 = sub_22077B0(1312);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 24) = 6;
    v4 = v2 + 568;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)v2 = &unk_49EDE80;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4F9E3BA;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 160) = &unk_49EDA68;
    *(_QWORD *)(v2 + 184) = v2 + 200;
    *(_QWORD *)(v2 + 416) = v2 + 432;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 168) = 0;
    *(_QWORD *)(v2 + 176) = 0;
    *(_QWORD *)(v2 + 192) = 0x1000000000LL;
    *(_QWORD *)(v2 + 424) = 0x1000000000LL;
    *(_DWORD *)(v2 + 560) = 0;
    *(_QWORD *)(v2 + 328) = 0;
    *(_QWORD *)(v2 + 336) = 0;
    *(_QWORD *)(v2 + 344) = 0;
    *(_QWORD *)(v2 + 352) = 0;
    *(_QWORD *)(v2 + 360) = 0;
    *(_QWORD *)(v2 + 368) = 0;
    *(_QWORD *)(v2 + 376) = 0;
    *(_QWORD *)(v2 + 384) = 1;
    *(_QWORD *)(v2 + 392) = 0;
    *(_QWORD *)(v2 + 400) = 0;
    *(_DWORD *)(v2 + 408) = 0;
    v5 = sub_22077B0(568);
    v6 = v5;
    if ( v5 )
    {
      *(_QWORD *)(v5 + 8) = 0;
      *(_DWORD *)(v5 + 24) = 5;
      *(_QWORD *)(v5 + 16) = &unk_4F9E389;
      *(_QWORD *)(v5 + 80) = v5 + 64;
      *(_QWORD *)(v5 + 88) = v5 + 64;
      *(_QWORD *)(v5 + 128) = v5 + 112;
      *(_QWORD *)(v5 + 136) = v5 + 112;
      *(_QWORD *)(v5 + 184) = v5 + 200;
      v6 = v5 + 160;
      *(_QWORD *)(v5 + 416) = v5 + 432;
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_QWORD *)(v5 + 48) = 0;
      *(_DWORD *)(v5 + 64) = 0;
      *(_QWORD *)(v5 + 72) = 0;
      *(_QWORD *)(v5 + 96) = 0;
      *(_DWORD *)(v5 + 112) = 0;
      *(_QWORD *)(v5 + 120) = 0;
      *(_QWORD *)(v5 + 144) = 0;
      *(_BYTE *)(v5 + 152) = 0;
      *(_QWORD *)(v5 + 168) = 0;
      *(_QWORD *)(v5 + 176) = 0;
      *(_QWORD *)(v5 + 192) = 0x1000000000LL;
      *(_QWORD *)(v5 + 424) = 0x1000000000LL;
      *(_DWORD *)(v5 + 560) = 0;
      *(_QWORD *)(v5 + 328) = 0;
      *(_QWORD *)(v5 + 336) = 0;
      *(_QWORD *)(v5 + 344) = 0;
      *(_QWORD *)(v5 + 352) = 0;
      *(_QWORD *)(v5 + 360) = 0;
      *(_QWORD *)(v5 + 368) = 0;
      *(_QWORD *)(v5 + 376) = 0;
      *(_QWORD *)(v5 + 384) = 1;
      *(_QWORD *)(v5 + 392) = 0;
      *(_QWORD *)(v5 + 400) = 0;
      *(_DWORD *)(v5 + 408) = 0;
      *(_QWORD *)v5 = &unk_49EDC10;
      *(_QWORD *)(v5 + 160) = &unk_49EDCC8;
    }
    sub_1611190(v3 + 568, v6);
    a1[1] = v3;
    *(_BYTE *)(v3 + 1304) = 0;
    *(_QWORD *)v3 = &unk_49ED7E8;
    *(_QWORD *)(v3 + 160) = &unk_49ED8A0;
    *(_QWORD *)(v3 + 568) = &unk_49ED8E0;
  }
  else
  {
    a1[1] = 0;
    v4 = 0;
  }
  *(_QWORD *)(v3 + 176) = v4;
  v7 = (_QWORD *)sub_22077B0(32);
  v8 = a1[1];
  if ( v7 )
  {
    *v7 = 0;
    v7[1] = 0;
    v7[2] = 0;
    v7[3] = v8 + 160;
  }
  return sub_1636870(v8, v7);
}
