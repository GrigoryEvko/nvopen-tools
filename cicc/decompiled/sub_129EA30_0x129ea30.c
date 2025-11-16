// Function: sub_129EA30
// Address: 0x129ea30
//
__int64 __fastcall sub_129EA30(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax

  v3 = a1 + 16;
  *(_QWORD *)(v3 - 16) = a2;
  v4 = *a2;
  sub_15A5590(v3, v4, 1, 0);
  *(_WORD *)(a1 + 492) = 0;
  *(_DWORD *)(a1 + 480) = 0;
  *(_WORD *)(a1 + 484) = 0;
  *(_DWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 504) = 8;
  v5 = sub_22077B0(64);
  v6 = *(_QWORD *)(a1 + 504);
  *(_QWORD *)(a1 + 496) = v5;
  v7 = (__int64 *)(v5 + ((4 * v6 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v8 = sub_22077B0(512);
  *(_QWORD *)(a1 + 536) = v7;
  *(_QWORD *)(a1 + 568) = v7;
  *(_QWORD *)(a1 + 528) = v8 + 512;
  *(_QWORD *)(a1 + 560) = v8 + 512;
  *v7 = v8;
  *(_QWORD *)(a1 + 520) = v8;
  *(_QWORD *)(a1 + 552) = v8;
  *(_QWORD *)(a1 + 512) = v8;
  *(_QWORD *)(a1 + 544) = v8;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_DWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = sub_8237A0(1024, v4, v8 + 512, v9, v10, v11);
  sub_7461E0(a1 + 648);
  *(_BYTE *)(a1 + 793) = 0;
  *(_QWORD *)(a1 + 648) = sub_729390;
  *(_QWORD *)(a1 + 664) = *(_QWORD *)(a1 + 640);
  result = sub_129E860(a1);
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
