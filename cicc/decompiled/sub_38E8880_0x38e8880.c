// Function: sub_38E8880
// Address: 0x38e8880
//
__int64 __fastcall sub_38E8880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdi
  __int64 v22; // rax
  __int64 v23; // r8

  v8 = sub_22077B0(0x370u);
  v9 = v8;
  if ( v8 )
  {
    sub_3909370(v8);
    *(_QWORD *)v9 = off_49D9310;
    sub_392A6B0(v9 + 144, a4);
    *(_QWORD *)(v9 + 320) = a2;
    if ( !a5 )
      a5 = 1;
    *(_QWORD *)(v9 + 344) = a1;
    *(_WORD *)(v9 + 384) = 0;
    *(_DWORD *)(v9 + 376) = a5;
    *(_QWORD *)(v9 + 328) = a3;
    *(_QWORD *)(v9 + 336) = a4;
    *(_QWORD *)(v9 + 368) = 0;
    *(_DWORD *)(v9 + 380) = 0;
    *(_QWORD *)(v9 + 392) = 0;
    *(_QWORD *)(v9 + 400) = 0;
    *(_QWORD *)(v9 + 408) = 0;
    *(_QWORD *)(v9 + 416) = 0;
    *(_QWORD *)(v9 + 424) = 0;
    *(_QWORD *)(v9 + 432) = 0x1800000000LL;
    *(_QWORD *)(v9 + 448) = 0;
    *(_QWORD *)(v9 + 456) = 0;
    *(_QWORD *)(v9 + 464) = 0;
    *(_QWORD *)(v9 + 472) = 0;
    *(_QWORD *)(v9 + 488) = 0;
    *(_QWORD *)(v9 + 496) = 0;
    *(_QWORD *)(v9 + 504) = 0;
    *(_QWORD *)(v9 + 512) = 0;
    *(_QWORD *)(v9 + 520) = 0;
    *(_QWORD *)(v9 + 528) = 0;
    *(_QWORD *)(v9 + 536) = 0;
    *(_QWORD *)(v9 + 544) = 0;
    *(_QWORD *)(v9 + 480) = 8;
    v10 = sub_22077B0(0x40u);
    v11 = *(_QWORD *)(v9 + 480);
    *(_QWORD *)(v9 + 472) = v10;
    v12 = (__int64 *)(v10 + ((4 * v11 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v13 = sub_22077B0(0x1F8u);
    *(_BYTE *)(v9 + 552) |= 1u;
    *v12 = v13;
    v14 = v13 + 504;
    *(_QWORD *)(v9 + 496) = v13;
    *(_QWORD *)(v9 + 528) = v13;
    *(_QWORD *)(v9 + 488) = v13;
    *(_QWORD *)(v9 + 520) = v13;
    *(_QWORD *)(v9 + 600) = v9 + 616;
    *(_QWORD *)(v9 + 608) = 0x400000000LL;
    *(_QWORD *)(v9 + 864) = 0x1000000000LL;
    v15 = *(_QWORD *)(v9 + 344);
    *(_QWORD *)(v9 + 504) = v14;
    *(_QWORD *)(v9 + 536) = v14;
    *(_QWORD *)(v9 + 512) = v12;
    *(_QWORD *)(v9 + 544) = v12;
    *(_QWORD *)(v9 + 560) = 0;
    *(_QWORD *)(v9 + 568) = 0;
    *(_QWORD *)(v9 + 576) = 0;
    *(_QWORD *)(v9 + 584) = 0;
    *(_DWORD *)(v9 + 592) = 0;
    *(_DWORD *)(v9 + 840) = -1;
    *(_WORD *)(v9 + 844) = 0;
    *(_BYTE *)(v9 + 846) = 0;
    *(_QWORD *)(v9 + 848) = 0;
    *(_QWORD *)(v9 + 856) = 0;
    *(_BYTE *)(v9 + 17) = 0;
    *(_QWORD *)(v9 + 352) = *(_QWORD *)(v15 + 48);
    *(_QWORD *)(v9 + 360) = *(_QWORD *)(v15 + 56);
    *(_QWORD *)(v15 + 48) = sub_38E5130;
    *(_QWORD *)(v15 + 56) = v9;
    v16 = *(_QWORD *)(**(_QWORD **)(v9 + 344) + 24LL * (unsigned int)(*(_DWORD *)(v9 + 376) - 1));
    sub_392A730(v9 + 144, *(_QWORD *)(v16 + 8), *(_QWORD *)(v16 + 16) - *(_QWORD *)(v16 + 8), 0);
    v17 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 680LL);
    if ( v17 == 2 )
    {
      v18 = sub_3901A20();
    }
    else
    {
      if ( v17 > 2 )
      {
        if ( v17 != 3 )
        {
          v20 = *(_QWORD *)(v9 + 368);
LABEL_10:
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 16LL))(v20, v9);
          sub_38E5A40(v9);
          *(_DWORD *)(v9 + 556) = 0;
          return v9;
        }
      }
      else if ( !v17 )
      {
        v22 = sub_39052D0();
        v23 = *(_QWORD *)(v9 + 368);
        v20 = v22;
        *(_QWORD *)(v9 + 368) = v22;
        if ( v23 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
          v20 = *(_QWORD *)(v9 + 368);
        }
        *(_BYTE *)(v9 + 844) = 1;
        goto LABEL_10;
      }
      v18 = sub_3907790();
    }
    v19 = *(_QWORD *)(v9 + 368);
    v20 = v18;
    *(_QWORD *)(v9 + 368) = v18;
    if ( v19 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
      v20 = *(_QWORD *)(v9 + 368);
    }
    goto LABEL_10;
  }
  return v9;
}
