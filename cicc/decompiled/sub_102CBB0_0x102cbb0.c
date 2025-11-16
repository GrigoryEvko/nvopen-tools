// Function: sub_102CBB0
// Address: 0x102cbb0
//
__int64 __fastcall sub_102CBB0(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v12 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v13 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = *a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v9 = v7 + 8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
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
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 344) = 0x400000000LL;
  *(_QWORD *)(a1 + 384) = a1 + 400;
  *(_QWORD *)(a1 + 256) = v6 + 8;
  *(_QWORD *)(a1 + 264) = v12 + 8;
  *(_QWORD *)(a1 + 272) = v13 + 8;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_DWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 1;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_DWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 1;
  *(_QWORD *)(a1 + 280) = v7 + 8;
  *(_QWORD *)(a1 + 416) = &unk_49DDC10;
  v10 = (_QWORD *)(a1 + 520);
  *(_QWORD *)(a1 + 424) = v9;
  do
  {
    if ( v10 )
      *v10 = -4096;
    v10 += 11;
  }
  while ( (_QWORD *)(a1 + 872) != v10 );
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = a1 + 904;
  *(_QWORD *)(a1 + 888) = 8;
  *(_DWORD *)(a1 + 896) = 0;
  *(_BYTE *)(a1 + 900) = 1;
  *(_BYTE *)(a1 + 968) = 0;
  *(_DWORD *)(a1 + 976) = v8;
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_DWORD *)(a1 + 1008) = 0;
  return a1;
}
