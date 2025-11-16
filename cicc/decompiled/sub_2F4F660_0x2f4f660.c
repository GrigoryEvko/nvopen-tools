// Function: sub_2F4F660
// Address: 0x2f4f660
//
__int64 __fastcall sub_2F4F660(__int64 a1, __int64 *a2, __int64 a3)
{
  void (__fastcall *v3)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v5)(__int64, _BYTE *, __int64); // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 result; // rax
  _BYTE v12[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v13)(__int64, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  __int64 v14; // [rsp+18h] [rbp-28h]

  v3 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
  v13 = 0;
  if ( v3 )
  {
    v3(v12, a3, 2);
    v14 = *(_QWORD *)(a3 + 24);
    v13 = *(void (__fastcall **)(__int64, _BYTE *, __int64))(a3 + 16);
  }
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_4A3A030;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  sub_2F5FEE0(a1 + 48);
  v5 = v13;
  *(_QWORD *)(a1 + 384) = 0;
  if ( v5 )
  {
    v5(a1 + 368, v12, 2);
    *(_QWORD *)(a1 + 392) = v14;
    v5 = v13;
    *(_QWORD *)(a1 + 384) = v13;
  }
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = a1 + 432;
  *(_QWORD *)(a1 + 688) = a1 + 704;
  *(_QWORD *)(a1 + 416) = 32;
  *(_DWORD *)(a1 + 424) = 0;
  *(_BYTE *)(a1 + 428) = 1;
  *(_QWORD *)(a1 + 696) = 0x200000000LL;
  *(_DWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = a1 + 720;
  *(_QWORD *)(a1 + 744) = a1 + 720;
  *(_QWORD *)(a1 + 752) = 0;
  if ( v5 )
    v5((__int64)v12, v12, 3);
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_BYTE *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 976) = 0;
  *(_BYTE *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  *(_QWORD *)(a1 + 1040) = 0;
  *(_DWORD *)(a1 + 1048) = 0;
  *(_QWORD *)a1 = off_4A2B1A8;
  *(_QWORD *)(a1 + 760) = &unk_4A2B218;
  v6 = a1 + 1056;
  do
  {
    *(_DWORD *)v6 = 0;
    *(_QWORD *)(v6 + 48) = v6 + 64;
    v7 = v6 + 528;
    v6 += 720;
    *(_DWORD *)(v6 - 716) = 0;
    *(_DWORD *)(v6 - 712) = 0;
    *(_QWORD *)(v6 - 704) = 0;
    *(_QWORD *)(v6 - 696) = 0;
    *(_QWORD *)(v6 - 688) = 0;
    *(_QWORD *)(v6 - 680) = 0;
    *(_DWORD *)(v6 - 664) = 0;
    *(_DWORD *)(v6 - 660) = 4;
    *(_QWORD *)(v6 - 208) = v7;
    *(_DWORD *)(v6 - 200) = 0;
    *(_DWORD *)(v6 - 196) = 8;
  }
  while ( v6 != a1 + 24096 );
  *(_QWORD *)(a1 + 28944) = 0;
  *(_QWORD *)(a1 + 24096) = a1 + 24112;
  *(_QWORD *)(a1 + 24176) = a1 + 24192;
  *(_QWORD *)(a1 + 24104) = 0x800000000LL;
  *(_QWORD *)(a1 + 24184) = 0x2000000000LL;
  *(_QWORD *)(a1 + 28808) = 0x2000000000LL;
  *(_QWORD *)(a1 + 28992) = 0x800000000LL;
  *(_QWORD *)(a1 + 28800) = a1 + 28816;
  *(_QWORD *)(a1 + 28952) = 0;
  *(_QWORD *)(a1 + 28984) = a1 + 29000;
  *(_QWORD *)(a1 + 29064) = 0;
  *(_QWORD *)(a1 + 29072) = 0;
  *(_WORD *)(a1 + 29080) = 0;
  v8 = *a2;
  *(_QWORD *)(a1 + 28960) = 0;
  *(_QWORD *)(a1 + 24) = v8;
  v9 = a2[1];
  *(_QWORD *)(a1 + 28968) = 0;
  *(_QWORD *)(a1 + 32) = v9;
  v10 = a2[2];
  *(_DWORD *)(a1 + 28976) = 0;
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 784) = a2[3];
  *(_QWORD *)(a1 + 792) = a2[4];
  *(_QWORD *)(a1 + 800) = a2[5];
  *(_QWORD *)(a1 + 808) = a2[6];
  *(_QWORD *)(a1 + 816) = a2[7];
  *(_QWORD *)(a1 + 824) = a2[8];
  *(_QWORD *)(a1 + 832) = a2[9];
  *(_QWORD *)(a1 + 840) = a2[10];
  *(_QWORD *)(a1 + 848) = a2[11];
  *(_QWORD *)(a1 + 856) = a2[12];
  result = a2[13];
  *(_QWORD *)(a1 + 864) = result;
  return result;
}
