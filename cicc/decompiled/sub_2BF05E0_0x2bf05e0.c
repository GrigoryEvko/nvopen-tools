// Function: sub_2BF05E0
// Address: 0x2bf05e0
//
__int64 __fastcall sub_2BF05E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 *a11)
{
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi

  v11 = (_QWORD *)(a1 + 136);
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 1;
  *(_QWORD *)(a1 + 8) = a3;
  do
  {
    if ( v11 )
      *v11 = -4096;
    v11 += 2;
  }
  while ( (_QWORD *)(a1 + 200) != v11 );
  *(_QWORD *)(a1 + 904) = a7;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x1000000000LL;
  *(_QWORD *)(a1 + 776) = a1 + 800;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = a6;
  *(_QWORD *)(a1 + 752) = 0;
  *(_BYTE *)(a1 + 760) = 1;
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 784) = 8;
  *(_DWORD *)(a1 + 792) = 0;
  *(_BYTE *)(a1 + 796) = 1;
  *(_WORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = a5;
  *(_QWORD *)(a1 + 912) = a8;
  *(_QWORD *)(a1 + 920) = a9;
  *(_QWORD *)(a1 + 928) = a10;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_DWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_DWORD *)(a1 + 1000) = 0;
  v12 = *a11;
  *(_QWORD *)(a1 + 1008) = a11;
  *(_QWORD *)(a1 + 1016) = v12;
  *(_QWORD *)(a1 + 1024) = a1 + 1040;
  *(_QWORD *)(a1 + 1032) = 0x100000000LL;
  v13 = a1 + 1064;
  v14 = a1 + 1024;
  *(_QWORD *)(v14 + 24) = v13;
  *(_QWORD *)(v14 + 32) = 0x600000000LL;
  *(_QWORD *)(v14 + 88) = 0;
  *(_QWORD *)(v14 + 96) = 0;
  *(_QWORD *)(v14 + 104) = 0;
  *(_DWORD *)(v14 + 112) = 0;
  *(_QWORD *)(v14 + 120) = 0;
  *(_BYTE *)(v14 + 136) = 0;
  *(_QWORD *)(v14 + 140) = 0;
  *(_QWORD *)(v14 + 128) = a9;
  return sub_2C06B20();
}
