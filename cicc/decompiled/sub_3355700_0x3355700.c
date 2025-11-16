// Function: sub_3355700
// Address: 0x3355700
//
__int64 __fastcall sub_3355700(__int64 a1)
{
  __int64 v1; // r15
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v15; // r14
  __int64 (*v16)(); // rax
  __int64 v17; // rax

  v1 = 0;
  v3 = *(__int64 **)(*(_QWORD *)(a1 + 40) + 16LL);
  v4 = *v3;
  v5 = *(__int64 (**)(void))(*v3 + 128);
  if ( v5 != sub_2DAC790 )
  {
    v1 = v5();
    v4 = *v3;
  }
  v6 = (*(__int64 (**)(void))(v4 + 200))();
  v7 = sub_22077B0(0xB0u);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    *(_BYTE *)(v7 + 12) = 0;
    *(_DWORD *)(v7 + 8) = 0;
    *(_QWORD *)(v7 + 56) = v8;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 40) = 0;
    *(_WORD *)(v7 + 44) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 64) = v1;
    *(_QWORD *)(v7 + 72) = v6;
    *(_QWORD *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 88) = 0;
    *(_QWORD *)(v7 + 96) = 0;
    *(_QWORD *)(v7 + 104) = 0;
    *(_QWORD *)(v7 + 112) = 0;
    *(_QWORD *)(v7 + 120) = 0;
    *(_QWORD *)(v7 + 128) = 0;
    *(_QWORD *)(v7 + 136) = 0;
    *(_QWORD *)(v7 + 144) = 0;
    *(_QWORD *)(v7 + 152) = 0;
    *(_QWORD *)(v7 + 160) = 0;
    *(_QWORD *)v7 = off_4A36358;
    *(_QWORD *)(v7 + 168) = v7;
  }
  v9 = sub_22077B0(0x5C8u);
  v10 = v9;
  if ( v9 )
  {
    v11 = *(_QWORD *)(a1 + 40);
    sub_335DCC0(v9, v11);
    *(_BYTE *)(v10 + 632) = 0;
    *(_QWORD *)v10 = off_4A36238;
    *(_QWORD *)(v10 + 680) = 0xFFFFFFFF00000000LL;
    *(_QWORD *)(v10 + 712) = v10 + 728;
    *(_QWORD *)(v10 + 720) = 0x400000000LL;
    *(_QWORD *)(v10 + 640) = v7;
    *(_QWORD *)(v10 + 648) = 0;
    *(_QWORD *)(v10 + 656) = 0;
    *(_QWORD *)(v10 + 664) = 0;
    *(_QWORD *)(v10 + 688) = 0;
    *(_QWORD *)(v10 + 696) = 0;
    *(_QWORD *)(v10 + 704) = 0;
    *(_QWORD *)(v10 + 760) = 0;
    *(_QWORD *)(v10 + 768) = 0;
    *(_QWORD *)(v10 + 776) = 0;
    *(_DWORD *)(v10 + 784) = 0;
    sub_2F8FF00(v10 + 792, v10 + 48, 0);
    v12 = (_QWORD *)(v10 + 1224);
    *(_QWORD *)(v10 + 1208) = 0;
    *(_QWORD *)(v10 + 1216) = 1;
    do
    {
      if ( v12 )
        *v12 = -4096;
      v12 += 2;
    }
    while ( (_QWORD *)(v10 + 1480) != v12 );
    if ( !byte_5038F08 && *(_BYTE *)(v10 + 632) )
    {
      v15 = *(_QWORD *)(v11 + 16);
      v16 = *(__int64 (**)())(*(_QWORD *)v15 + 128LL);
      if ( v16 == sub_2DAC790 )
        BUG();
      v17 = ((__int64 (__fastcall *)(__int64))v16)(v15);
      *(_QWORD *)(v10 + 672) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v17 + 1032LL))(
                                 v17,
                                 v15,
                                 v10);
    }
    else
    {
      v13 = sub_22077B0(0x10u);
      if ( v13 )
      {
        *(_DWORD *)(v13 + 8) = 0;
        *(_QWORD *)v13 = &unk_4A2BCD8;
      }
      *(_QWORD *)(v10 + 672) = v13;
    }
  }
  *(_QWORD *)(v7 + 88) = v10;
  return v10;
}
