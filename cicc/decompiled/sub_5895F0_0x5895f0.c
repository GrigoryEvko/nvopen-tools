// Function: sub_5895F0
// Address: 0x5895f0
//
__int64 __fastcall sub_5895F0(__int64 a1, int a2)
{
  int v2; // edx
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r14
  _QWORD *v8; // rsi
  __int64 *v9; // r14
  unsigned __int64 v10; // rbx
  __int64 v11; // rdi
  _QWORD v13[2]; // [rsp+10h] [rbp-7D0h] BYREF
  char v14; // [rsp+20h] [rbp-7C0h] BYREF
  _QWORD v15[6]; // [rsp+30h] [rbp-7B0h] BYREF
  _QWORD v16[8]; // [rsp+60h] [rbp-780h] BYREF
  _BYTE v17[64]; // [rsp+A0h] [rbp-740h] BYREF
  _QWORD *v18; // [rsp+E0h] [rbp-700h] BYREF
  _QWORD v19[5]; // [rsp+E8h] [rbp-6F8h] BYREF
  int v20; // [rsp+110h] [rbp-6D0h]
  int v21; // [rsp+118h] [rbp-6C8h]
  __int64 v22; // [rsp+120h] [rbp-6C0h]
  _BYTE *v23; // [rsp+130h] [rbp-6B0h]
  __int128 v24; // [rsp+138h] [rbp-6A8h]
  __int128 v25; // [rsp+148h] [rbp-698h]
  __int128 v26; // [rsp+158h] [rbp-688h]
  __int128 v27; // [rsp+168h] [rbp-678h]
  __int64 v28; // [rsp+178h] [rbp-668h]
  _BYTE v29[224]; // [rsp+180h] [rbp-660h] BYREF
  _QWORD v30[22]; // [rsp+260h] [rbp-580h] BYREF
  void *v31; // [rsp+310h] [rbp-4D0h] BYREF
  __int64 v32; // [rsp+498h] [rbp-348h]
  __int64 v33; // [rsp+4A0h] [rbp-340h]
  __int64 *v34; // [rsp+4C0h] [rbp-320h]
  __int64 v35; // [rsp+4E0h] [rbp-300h]
  _QWORD v36[2]; // [rsp+500h] [rbp-2E0h] BYREF
  void (__fastcall *v37)(_QWORD *, _QWORD *, __int64); // [rsp+510h] [rbp-2D0h]
  __int64 v38; // [rsp+518h] [rbp-2C8h]
  __int64 v39; // [rsp+520h] [rbp-2C0h]
  unsigned int v40; // [rsp+528h] [rbp-2B8h]
  char *v41; // [rsp+678h] [rbp-168h]
  char v42; // [rsp+688h] [rbp-158h] BYREF
  void *v43; // [rsp+710h] [rbp-D0h] BYREF
  char v44; // [rsp+720h] [rbp-C0h]
  __int64 v45; // [rsp+728h] [rbp-B8h]
  unsigned int v46; // [rsp+730h] [rbp-B0h]

  sub_271ED00();
  sub_2D45FB0();
  sub_D05620();
  sub_DF47B0();
  sub_E00710();
  sub_DF51B0();
  sub_F42540();
  sub_2285ED0();
  sub_2285DF0();
  v30[0] = 0x100000000000001LL;
  v30[1] = 0x1000101000000LL;
  v30[2] = 0;
  v37 = 0;
  sub_2975000((unsigned int)v36, a2, v2, 0x1000000, v3, v4, 0x100000000000001LL, 0x1000101000000LL, 0);
  if ( v37 )
    v37(v36, v36, 3);
  sub_298B2A0(0);
  sub_22A9C30();
  sub_22A9A60();
  sub_25AF8E0();
  sub_2752860();
  sub_228CBE0();
  sub_229CC00();
  sub_229CAB0();
  sub_229CEA0();
  sub_229CD50();
  sub_24FB600(1);
  sub_2DDD300();
  sub_D1DA60();
  sub_27D88E0();
  sub_F11260();
  sub_2DF2740();
  sub_2DF4550();
  sub_11CE100();
  sub_27EE150(0);
  sub_22C1200();
  sub_26184C0();
  sub_F68080();
  sub_285D3E0();
  sub_287C030();
  sub_28810B0(2, 0, 0, -1, -1, -1, -1, -1, -1);
  sub_2A2E410();
  sub_2A309A0();
  sub_2A32EF0(0);
  sub_28C1B50();
  sub_270B4D0();
  sub_2A36D10();
  sub_28FE960();
  sub_229D140();
  sub_229CFF0();
  sub_229D3E0();
  sub_229D290();
  sub_28EE4F0();
  sub_22DC4E0();
  sub_22E5B20();
  sub_22E5DC0();
  sub_22E59D0();
  sub_22E5C70();
  sub_2F84270();
  sub_291E7B0(1);
  sub_26185F0();
  sub_2998330();
  sub_2730460();
  sub_2D5CDB0();
  sub_29CE7D0();
  sub_277BAB0(0);
  sub_278CB40();
  sub_104C740();
  sub_28BB450();
  sub_2DBA350();
  sub_2DC2F80();
  v13[0] = &v14;
  v16[6] = v13;
  v16[0] = &unk_49DD210;
  v13[1] = 0;
  v14 = 0;
  memset(&v16[1], 0, 32);
  v16[5] = 0x100000000LL;
  sub_CB5980(v16, 0, 0, 0);
  sub_A758C0(v36, byte_3F871B3, v30);
  sub_B3A980(v16, v36, 0);
  sub_2240A30(v36);
  sub_A758C0(v36, byte_3F871B3, v30);
  sub_B3AAD0(v16, v36);
  sub_2240A30(v36);
  sub_2977E10(0);
  sub_2891640();
  sub_2A8DB50();
  sub_28E4630();
  v36[0] = 0x100000000LL;
  v36[1] = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  sub_293BA20(v36);
  sub_C7D6A0(v37, 4LL * (unsigned int)v39, 4);
  sub_2952040(0);
  sub_297B310();
  sub_297B570();
  sub_297E780();
  sub_2946730();
  sub_2DE21C0();
  sub_2A86B80();
  sub_29D3980();
  sub_2F9C2C0();
  v5 = sub_22077B0(184);
  if ( v5 )
    sub_D9A9E0(v5);
  LOWORD(v39) = 257;
  v6 = sub_BD2DA0(136);
  v7 = v6;
  if ( v6 )
    sub_B2C3B0(v6, 0, 0, 0xFFFFFFFFLL, v36, 0);
  sub_11FAD30(v7);
  sub_22E3560(v30);
  sub_97F3E0(v29);
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v23 = v29;
  v28 = 0;
  sub_CF4B40(v17);
  sub_226B340(v36, v17);
  v18 = v36;
  v8 = v15;
  v19[1] = v19;
  memset(&v19[2], 0, 24);
  v19[0] = (char *)v19 + 4;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v15[0] = 0;
  v15[1] = -1;
  memset(&v15[2], 0, 32);
  sub_FD9690(&v18, v15);
  sub_C92250();
  sub_23CE240();
  sub_226B440(&v18);
  v43 = &unk_49DDBE8;
  if ( (v44 & 1) == 0 )
  {
    v8 = (_QWORD *)(16LL * v46);
    sub_C7D6A0(v45, v8, 8);
  }
  nullsub_184(&v43);
  if ( v41 != &v42 )
    _libc_free(v41, v8);
  if ( (v38 & 1) == 0 )
    sub_C7D6A0(v39, 40LL * v40, 8);
  sub_CF4BF0(v17);
  sub_226B620(v29);
  v30[0] = &unk_4A0A190;
  v31 = &unk_4A0A248;
  if ( v32 )
  {
    v9 = v34;
    v10 = v35 + 8;
    while ( v10 > (unsigned __int64)v9 )
    {
      v11 = *v9++;
      j_j___libc_free_0(v11, 512);
    }
    j_j___libc_free_0(v32, 8 * v33);
  }
  sub_B81E70(&v31);
  v30[0] = &unk_49DAF80;
  sub_BB9100(v30);
  v16[0] = &unk_49DD210;
  sub_CB5840(v16);
  return sub_2240A30(v13);
}
