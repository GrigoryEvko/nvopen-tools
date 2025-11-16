// Function: sub_35B6100
// Address: 0x35b6100
//
__int64 __fastcall sub_35B6100(__int64 a1, __int64 a2)
{
  void (__fastcall *v2)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v3)(__int64, _BYTE *, __int64); // rax
  _BYTE v5[16]; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v6)(__int64, _BYTE *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v7; // [rsp+18h] [rbp-18h]

  *(_QWORD *)(a1 + 16) = &unk_503FDCC;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
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
  *(_QWORD *)a1 = &unk_4A28EF0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  v6 = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  v2 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a2 + 16);
  if ( v2 )
  {
    v2(v5, a2, 2);
    v7 = *(_QWORD *)(a2 + 24);
    v6 = *(void (__fastcall **)(__int64, _BYTE *, __int64))(a2 + 16);
  }
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 200) = &unk_4A3A030;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  sub_2F5FEE0(a1 + 248);
  v3 = v6;
  *(_QWORD *)(a1 + 584) = 0;
  if ( v3 )
  {
    v3(a1 + 568, v5, 2);
    *(_QWORD *)(a1 + 592) = v7;
    v3 = v6;
    *(_QWORD *)(a1 + 584) = v6;
  }
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = a1 + 632;
  *(_QWORD *)(a1 + 888) = a1 + 904;
  *(_QWORD *)(a1 + 616) = 32;
  *(_DWORD *)(a1 + 624) = 0;
  *(_BYTE *)(a1 + 628) = 1;
  *(_QWORD *)(a1 + 896) = 0x200000000LL;
  *(_DWORD *)(a1 + 920) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 936) = a1 + 920;
  *(_QWORD *)(a1 + 944) = a1 + 920;
  *(_QWORD *)(a1 + 952) = 0;
  if ( v3 )
    v3((__int64)v5, v5, 3);
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)a1 = off_4A3A088;
  *(_QWORD *)(a1 + 200) = &unk_4A3A180;
  *(_QWORD *)(a1 + 960) = &unk_4A3A1D8;
  *(_QWORD *)(a1 + 1016) = a1 + 1032;
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 1024) = 0x600000000LL;
  *(_DWORD *)(a1 + 1080) = 0;
  return 0x600000000LL;
}
