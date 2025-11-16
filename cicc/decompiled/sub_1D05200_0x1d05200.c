// Function: sub_1D05200
// Address: 0x1d05200
//
__int64 __fastcall sub_1D05200(__int64 a1)
{
  __int64 v1; // r14
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 (*v4)(); // rdx
  __int64 (*v5)(); // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  bool v12; // zf
  __int64 v13; // rax
  __int64 v15; // r14
  __int64 (*v16)(); // rax
  __int64 v17; // rax

  v1 = 0;
  v2 = *(__int64 **)(*(_QWORD *)(a1 + 256) + 16LL);
  v3 = *v2;
  v4 = *(__int64 (**)())(*v2 + 40);
  if ( v4 != sub_1D00B00 )
  {
    v1 = ((__int64 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 16LL));
    v3 = *v2;
  }
  v5 = *(__int64 (**)())(v3 + 112);
  v6 = 0;
  if ( v5 != sub_1D00B10 )
    v6 = ((__int64 (__fastcall *)(__int64 *))v5)(v2);
  v7 = sub_22077B0(176);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 256);
    *(_DWORD *)(v7 + 8) = 0;
    *(_BYTE *)(v7 + 12) = 0;
    *(_QWORD *)(v7 + 56) = v8;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 40) = 0;
    *(_WORD *)(v7 + 44) = 0;
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
    *(_QWORD *)v7 = off_49F95B8;
    *(_QWORD *)(v7 + 168) = v7;
  }
  v9 = sub_22077B0(944);
  v10 = v9;
  if ( v9 )
  {
    v11 = *(_QWORD *)(a1 + 256);
    sub_1D0DA30(v9, v11);
    *(_BYTE *)(v10 + 664) = 0;
    *(_QWORD *)v10 = off_49F94A0;
    *(_QWORD *)(v10 + 744) = v10 + 760;
    *(_QWORD *)(v10 + 672) = v7;
    *(_QWORD *)(v10 + 680) = 0;
    *(_QWORD *)(v10 + 688) = 0;
    *(_QWORD *)(v10 + 696) = 0;
    *(_DWORD *)(v10 + 712) = 0;
    *(_QWORD *)(v10 + 728) = 0;
    *(_QWORD *)(v10 + 736) = 0;
    *(_QWORD *)(v10 + 752) = 0x400000000LL;
    *(_QWORD *)(v10 + 792) = 0;
    *(_QWORD *)(v10 + 800) = 0;
    *(_QWORD *)(v10 + 808) = 0;
    *(_DWORD *)(v10 + 816) = 0;
    sub_1F024C0(v10 + 824, v10 + 48, 0);
    v12 = byte_4FC13A0 == 0;
    *(_QWORD *)(v10 + 912) = 0;
    *(_QWORD *)(v10 + 920) = 0;
    *(_QWORD *)(v10 + 928) = 0;
    *(_DWORD *)(v10 + 936) = 0;
    if ( v12 && *(_BYTE *)(v10 + 664) )
    {
      v15 = *(_QWORD *)(v11 + 16);
      v16 = *(__int64 (**)())(*(_QWORD *)v15 + 40LL);
      if ( v16 == sub_1D00B00 )
        BUG();
      v17 = ((__int64 (__fastcall *)(_QWORD))v16)(*(_QWORD *)(v11 + 16));
      *(_QWORD *)(v10 + 704) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v17 + 752LL))(
                                 v17,
                                 v15,
                                 v10);
    }
    else
    {
      v13 = sub_22077B0(16);
      if ( v13 )
      {
        *(_DWORD *)(v13 + 8) = 0;
        *(_QWORD *)v13 = &unk_49FE598;
      }
      *(_QWORD *)(v10 + 704) = v13;
    }
  }
  *(_QWORD *)(v7 + 88) = v10;
  return v10;
}
