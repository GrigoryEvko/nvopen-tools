// Function: sub_31DA2B0
// Address: 0x31da2b0
//
__int64 __fastcall sub_31DA2B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 (*v10)(void); // rdx
  char v11; // al
  __int64 result; // rax

  *(_QWORD *)(a1 + 200) = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_5035D79;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
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
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)a1 = &unk_4A35060;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  v4 = *(_QWORD *)(a2 + 656);
  *(_QWORD *)(a1 + 208) = v4;
  v6 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
  *(_QWORD *)(a1 + 224) = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 216) = v6;
  *(_QWORD *)a3 = 0;
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 384) = a1 + 400;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_DWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 552) = a1 + 568;
  *(_QWORD *)(a1 + 560) = 0x100000000LL;
  v7 = a1 + 592;
  v8 = a1 + 616;
  *(_QWORD *)(v8 - 40) = v7;
  *(_QWORD *)(v8 - 32) = 0x200000000LL;
  *(_QWORD *)(v8 - 176) = 0;
  *(_QWORD *)(v8 - 168) = 0;
  *(_QWORD *)(v8 - 160) = 0;
  *(_QWORD *)(v8 - 152) = 0;
  *(_QWORD *)(v8 - 144) = 0;
  *(_DWORD *)(v8 - 136) = 0;
  *(_QWORD *)(v8 - 120) = 0;
  *(_QWORD *)(v8 - 112) = 0;
  *(_QWORD *)(v8 - 104) = 0;
  *(_QWORD *)(v8 - 96) = 0;
  *(_DWORD *)(v8 - 88) = 0;
  *(_QWORD *)(v8 - 80) = 0;
  *(_QWORD *)(v8 - 72) = 0;
  *(_QWORD *)(v8 - 8) = 0;
  sub_2FC8820(v8, a1);
  v9 = *(_QWORD *)(a1 + 224);
  *(_QWORD *)(a1 + 744) = 0;
  *(_WORD *)(a1 + 780) = 0;
  *(_QWORD *)(a1 + 784) = a1 + 800;
  *(_QWORD *)(a1 + 792) = 0x400000000LL;
  *(_QWORD *)(a1 + 752) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_DWORD *)(a1 + 776) = 0;
  *(_BYTE *)(a1 + 782) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0xFFFFFFFF00000000LL;
  *(_BYTE *)(a1 + 976) = 0;
  v10 = *(__int64 (**)(void))(*(_QWORD *)v9 + 96LL);
  v11 = 0;
  if ( v10 != sub_C13EE0 )
    v11 = v10();
  *(_BYTE *)(a1 + 488) = v11;
  result = *(unsigned __int8 *)(*(_QWORD *)(a1 + 208) + 348LL);
  *(_BYTE *)(a1 + 976) = result;
  return result;
}
