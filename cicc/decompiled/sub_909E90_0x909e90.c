// Function: sub_909E90
// Address: 0x909e90
//
__int64 __fastcall sub_909E90(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rax

  *(_QWORD *)a1 = a2;
  sub_917CE0();
  v5 = *a2;
  *(_QWORD *)(a1 + 352) = a3;
  v6 = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 472) = 0x1000000000LL;
  *(_QWORD *)(a1 + 496) = 0x1000000000LL;
  *(_QWORD *)(a1 + 528) = a1 + 512;
  *(_QWORD *)(a1 + 536) = a1 + 512;
  *(_QWORD *)(a1 + 576) = a1 + 560;
  *(_QWORD *)(a1 + 584) = a1 + 560;
  *(_QWORD *)(a1 + 624) = a1 + 608;
  *(_QWORD *)(a1 + 344) = v5;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_DWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 632) = a1 + 608;
  *(_QWORD *)(a1 + 672) = a1 + 656;
  *(_QWORD *)(a1 + 680) = a1 + 656;
  *(_BYTE *)(a1 + 360) |= 1u;
  *(_QWORD *)(a1 + 640) = 0;
  v7 = unk_4D04658;
  *(_DWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  if ( !v7 )
  {
    v15 = sub_22077B0(784);
    v6 = v15;
    if ( v15 )
      sub_93F4E0(v15, a1);
  }
  *(_QWORD *)(a1 + 368) = v6;
  v8 = ((__int64 (*)(void))sub_BCB2B0)();
  v9 = sub_BCE760(v8, 0);
  v10 = *(_QWORD *)(a1 + 344);
  *(_QWORD *)(a1 + 696) = v9;
  v11 = sub_BCB2E0(v10);
  v12 = *(_QWORD *)(a1 + 344);
  *(_QWORD *)(a1 + 704) = v11;
  v13 = sub_BCB2B0(v12);
  *(_QWORD *)(a1 + 712) = v13;
  result = sub_BCE770(v13, *(unsigned int *)(*(_QWORD *)(a1 + 352) + 4LL));
  *(_QWORD *)(a1 + 720) = result;
  return result;
}
