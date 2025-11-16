// Function: sub_40E313
// Address: 0x40e313
//
__int64 __fastcall sub_40E313(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  __int64 v8; // r10
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // r10
  __int64 v16; // rcx
  __int64 v17; // r11
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // rcx
  __int64 v21; // r11
  __int64 v22; // rcx
  __int64 v23; // r10
  __int64 v24; // rcx
  __int64 v25; // r11
  __int64 v26; // rcx
  __int64 v27; // r10
  __int64 v28; // rcx
  __int64 v29; // r11
  __int64 v30; // rcx
  __int64 v31; // r10
  __int64 v32; // r8
  __int64 v33; // r8
  __int64 v34; // rcx

  v6 = a1;
  if ( a3 )
  {
    sub_40E2CD(a3, v6);
    *(_DWORD *)(a3 + 8) = 9;
    *(_QWORD *)(a3 + 16) = v8;
    *(_QWORD *)a3 = 0x1500000000LL;
  }
  v9 = v6;
  sub_40E2CD(a4, v6);
  *(_QWORD *)v10 = v11;
  *(_QWORD *)(v10 + 16) = "n_lock_ops";
  *(_DWORD *)(v10 + 8) = 9;
  sub_40E2CD(v10 + 40, v9);
  *(_QWORD *)(v12 + 40) = 0x800000001LL;
  *(_DWORD *)(v12 + 48) = 9;
  *(_QWORD *)(v12 + 56) = v13;
  sub_40E2CD(v12 + 80, v9);
  *(_QWORD *)(v14 + 80) = v15;
  *(_QWORD *)(v14 + 96) = "n_waiting";
  *(_DWORD *)(v14 + 88) = 9;
  sub_40E2CD(v14 + 120, v9);
  *(_QWORD *)(v16 + 120) = 0x800000001LL;
  *(_DWORD *)(v16 + 128) = 9;
  *(_QWORD *)(v16 + 136) = v17;
  sub_40E2CD(v16 + 160, v9);
  *(_QWORD *)(v18 + 160) = v19;
  *(_QWORD *)(v18 + 176) = "n_spin_acq";
  *(_DWORD *)(v18 + 168) = 9;
  sub_40E2CD(v18 + 200, v9);
  *(_QWORD *)(v20 + 200) = 0x800000001LL;
  *(_DWORD *)(v20 + 208) = 9;
  *(_QWORD *)(v20 + 216) = v21;
  sub_40E2CD(v20 + 240, v9);
  *(_QWORD *)(v22 + 240) = v23;
  *(_QWORD *)(v22 + 256) = "n_owner_switch";
  *(_DWORD *)(v22 + 248) = 9;
  sub_40E2CD(v22 + 280, v9);
  *(_QWORD *)(v24 + 280) = 0x800000001LL;
  *(_DWORD *)(v24 + 288) = 9;
  *(_QWORD *)(v24 + 296) = v25;
  sub_40E2CD(v24 + 320, v9);
  *(_QWORD *)(v26 + 320) = v27;
  *(_QWORD *)(v26 + 336) = "total_wait_ns";
  *(_DWORD *)(v26 + 328) = 9;
  sub_40E2CD(v26 + 360, v9);
  *(_QWORD *)(v28 + 360) = 0x800000001LL;
  *(_DWORD *)(v28 + 368) = 9;
  *(_QWORD *)(v28 + 376) = v29;
  sub_40E2CD(v28 + 400, v9);
  *(_QWORD *)(v30 + 400) = v31;
  *(_QWORD *)(v30 + 416) = "max_wait_ns";
  *(_DWORD *)(v30 + 408) = 9;
  sub_40E2CD(v32, v9);
  *(_DWORD *)(v33 + 8) = 9;
  *(_QWORD *)v33 = 0xC00000001LL;
  *(_QWORD *)(v33 + 16) = "max_n_thds";
  *(_DWORD *)(v34 + 364) = 10;
  return a6;
}
