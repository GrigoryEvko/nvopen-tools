// Function: sub_2339E50
// Address: 0x2339e50
//
__int64 __fastcall sub_2339E50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD *v6; // rax

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_WORD *)(a1 + 12) = WORD2(a3);
  v4 = a1 + 96;
  v5 = a1 + 136;
  *(_QWORD *)(v5 - 56) = v4;
  *(_QWORD *)(v5 - 104) = 0;
  *(_QWORD *)(v5 - 96) = 0;
  *(_QWORD *)(v5 - 88) = 0;
  *(_QWORD *)(v5 - 80) = 0;
  *(_QWORD *)(v5 - 72) = 0;
  *(_DWORD *)(v5 - 64) = 0;
  *(_QWORD *)(v5 - 48) = 0;
  *(_QWORD *)(v5 - 40) = 0;
  *(_QWORD *)(v5 - 32) = 0;
  *(_QWORD *)(v5 - 24) = 0;
  *(_QWORD *)(v5 - 16) = 0;
  sub_278A360();
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 1;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 1;
  *(_QWORD *)(a1 + 408) = 0x400000000LL;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  v6 = (_QWORD *)(a1 + 504);
  do
  {
    if ( v6 )
      *v6 = -4096;
    v6 += 2;
  }
  while ( (_QWORD *)(a1 + 568) != v6 );
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 568) = a1 + 584;
  *(_QWORD *)(a1 + 648) = a1 + 664;
  *(_QWORD *)(a1 + 576) = 0x400000000LL;
  *(_QWORD *)(a1 + 656) = 0x800000000LL;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_DWORD *)(a1 + 752) = 0;
  *(_BYTE *)(a1 + 760) = 1;
  *(_QWORD *)(a1 + 768) = a1 + 784;
  *(_QWORD *)(a1 + 776) = 0x400000000LL;
  return 0x400000000LL;
}
