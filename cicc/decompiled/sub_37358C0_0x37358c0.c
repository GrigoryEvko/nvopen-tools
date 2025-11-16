// Function: sub_37358C0
// Address: 0x37358c0
//
__int64 __fastcall sub_37358C0(__int64 a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  unsigned __int16 v12; // ax
  __int16 v13; // si
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v19; // [rsp-10h] [rbp-80h]
  const char *v21; // [rsp+10h] [rbp-60h] BYREF
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]

  v12 = sub_3220AA0(a5);
  if ( a7 || (v13 = 74, v12 <= 4u) )
    v13 = 17;
  sub_32476C0(a1, v13, (__int64)a3, a4, a5, a6, a2);
  *(_BYTE *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)a1 = &unk_4A3D3C8;
  *(_QWORD *)(a1 + 440) = 0x1000000000LL;
  *(_QWORD *)(a1 + 464) = 0x1000000000LL;
  *(_QWORD *)(a1 + 472) = a1 + 488;
  *(_QWORD *)(a1 + 480) = 0x200000000LL;
  *(_QWORD *)(a1 + 536) = a1 + 560;
  *(_QWORD *)(a1 + 592) = a1 + 608;
  *(_QWORD *)(a1 + 600) = 0x400000000LL;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 544) = 4;
  *(_DWORD *)(a1 + 552) = 0;
  *(_BYTE *)(a1 + 556) = 1;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_DWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_DWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0;
  *(_QWORD *)(a1 + 744) = 0;
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  sub_324C3F0(a1, a3, a1 + 8);
  v14 = *(_QWORD *)(a1 + 184);
  v23 = 1;
  v21 = "cu_macro_begin";
  v22 = 3;
  *(_QWORD *)(a1 + 416) = sub_31DCC50(v14, (__int64 *)&v21, v15, v16, v17);
  return v19;
}
