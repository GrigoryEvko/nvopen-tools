// Function: sub_93F4E0
// Address: 0x93f4e0
//
__int64 __fastcall sub_93F4E0(__int64 a1, __int64 *a2)
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
  sub_AE0470(v3, v4, 1, 0);
  *(_WORD *)(a1 + 460) = 0;
  *(_DWORD *)(a1 + 448) = 0;
  *(_WORD *)(a1 + 452) = 0;
  *(_DWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 472) = 8;
  v5 = sub_22077B0(64);
  v6 = *(_QWORD *)(a1 + 472);
  *(_QWORD *)(a1 + 464) = v5;
  v7 = (__int64 *)(v5 + ((4 * v6 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v8 = sub_22077B0(512);
  *(_QWORD *)(a1 + 504) = v7;
  *(_QWORD *)(a1 + 536) = v7;
  *(_QWORD *)(a1 + 496) = v8 + 512;
  *(_QWORD *)(a1 + 528) = v8 + 512;
  *v7 = v8;
  *(_QWORD *)(a1 + 488) = v8;
  *(_QWORD *)(a1 + 520) = v8;
  *(_QWORD *)(a1 + 480) = v8;
  *(_QWORD *)(a1 + 512) = v8;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_DWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = sub_8237A0(1024, v4, v8 + 512, v9, v10, v11);
  sub_7461E0(a1 + 616);
  *(_BYTE *)(a1 + 761) = 0;
  *(_QWORD *)(a1 + 616) = sub_729390;
  *(_QWORD *)(a1 + 632) = *(_QWORD *)(a1 + 608);
  result = sub_93F2D0(a1);
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
