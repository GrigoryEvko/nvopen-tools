// Function: sub_1269840
// Address: 0x1269840
//
__int64 __fastcall sub_1269840(__int64 a1, __int64 *a2, __int64 a3)
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
  sub_12778D0();
  v5 = *a2;
  *(_QWORD *)(a1 + 368) = a3;
  v6 = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 488) = 0x1000000000LL;
  *(_QWORD *)(a1 + 520) = 0x1000000000LL;
  *(_QWORD *)(a1 + 560) = a1 + 544;
  *(_QWORD *)(a1 + 568) = a1 + 544;
  *(_QWORD *)(a1 + 608) = a1 + 592;
  *(_QWORD *)(a1 + 616) = a1 + 592;
  *(_QWORD *)(a1 + 656) = a1 + 640;
  *(_QWORD *)(a1 + 360) = v5;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_DWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_DWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 664) = a1 + 640;
  *(_QWORD *)(a1 + 704) = a1 + 688;
  *(_QWORD *)(a1 + 712) = a1 + 688;
  *(_BYTE *)(a1 + 376) |= 1u;
  *(_QWORD *)(a1 + 672) = 0;
  v7 = dword_4D04658;
  *(_DWORD *)(a1 + 688) = 0;
  *(_QWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  if ( !v7 )
  {
    v15 = sub_22077B0(816);
    v6 = v15;
    if ( v15 )
      sub_129EA30(v15, a1);
  }
  *(_QWORD *)(a1 + 384) = v6;
  v8 = ((__int64 (*)(void))sub_1643330)();
  v9 = sub_1646BA0(v8, 0);
  v10 = *(_QWORD *)(a1 + 360);
  *(_QWORD *)(a1 + 728) = v9;
  v11 = sub_1643360(v10);
  v12 = *(_QWORD *)(a1 + 360);
  *(_QWORD *)(a1 + 736) = v11;
  v13 = sub_1643330(v12);
  *(_QWORD *)(a1 + 744) = v13;
  result = sub_1647190(v13, *(unsigned int *)(*(_QWORD *)(a1 + 368) + 4LL));
  *(_QWORD *)(a1 + 752) = result;
  return result;
}
