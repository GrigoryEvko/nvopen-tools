// Function: sub_2695720
// Address: 0x2695720
//
void __fastcall sub_2695720(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, char a6)
{
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  void (__fastcall *v16)(__int64, _QWORD *, __int64); // rax
  void (__fastcall *v17)(__int64, _QWORD *, __int64); // rax
  void (__fastcall *v18)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  _OWORD *v21; // rax
  __int64 v22; // rdx
  __m128i si128; // xmm0
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rdi
  bool v30; // al
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  _QWORD **v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 **v39; // rax
  __int64 v40; // rax
  __int64 (__fastcall **v41)(__int64 *, __int64 *, int); // rdi
  _QWORD *v42; // [rsp+8h] [rbp-B8h]
  _QWORD v43[2]; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v44)(__int64, _QWORD *, __int64); // [rsp+20h] [rbp-A0h]
  __int64 (__fastcall *v45)(__int64 **, __int64); // [rsp+28h] [rbp-98h]
  _QWORD v46[2]; // [rsp+30h] [rbp-90h] BYREF
  void (__fastcall *v47)(__int64, _QWORD *, __int64); // [rsp+40h] [rbp-80h]
  __int64 (__fastcall *v48)(__int64 **, __int64); // [rsp+48h] [rbp-78h]
  __int64 v49[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 (__fastcall *v50)(__int64 *, __int64 *, int); // [rsp+60h] [rbp-60h] BYREF
  __int64 (__fastcall *v51)(__int64 **, __int64); // [rsp+68h] [rbp-58h]
  __int64 v52; // [rsp+70h] [rbp-50h]
  __int64 v53; // [rsp+78h] [rbp-48h]
  __int64 v54; // [rsp+80h] [rbp-40h]

  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 48) = 0x600000000LL;
  *(_QWORD *)(a1 + 104) = a2 + 312;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  v10 = a1 + 280;
  v11 = (__int64 *)(a1 + 344);
  *(v11 - 43) = a5;
  *(v11 - 29) = (__int64)a4;
  *(v11 - 13) = a3;
  *(v11 - 42) = 0;
  *(v11 - 41) = 0;
  *(v11 - 40) = 0;
  *((_DWORD *)v11 - 78) = 0;
  *(v11 - 28) = 0;
  *(v11 - 27) = 0;
  *(v11 - 26) = 0;
  *(v11 - 25) = 0;
  *((_DWORD *)v11 - 48) = 0;
  *(v11 - 23) = 0;
  *(v11 - 22) = 0;
  *(v11 - 21) = 0;
  *((_DWORD *)v11 - 40) = 0;
  *(v11 - 18) = 0;
  *(v11 - 17) = 0;
  *(v11 - 16) = 0;
  *(v11 - 15) = 0;
  *((_DWORD *)v11 - 28) = 0;
  *(v11 - 12) = 0;
  *(v11 - 11) = v10;
  *(v11 - 10) = 8;
  *((_DWORD *)v11 - 18) = 0;
  *((_BYTE *)v11 - 68) = 1;
  *(_QWORD *)(a1 + 344) = a1 + 360;
  sub_266F100(v11, *(_BYTE **)(a2 + 232), *(_QWORD *)(a2 + 232) + *(_QWORD *)(a2 + 240));
  v12 = *(_QWORD *)(a2 + 264);
  v49[0] = a3;
  v46[0] = a3;
  *(_QWORD *)(a1 + 376) = v12;
  v13 = *(_QWORD *)(a2 + 272);
  v43[0] = a3;
  *(_QWORD *)(a1 + 384) = v13;
  *(_QWORD *)(a1 + 392) = *(_QWORD *)(a2 + 280);
  v51 = sub_2506F90;
  v50 = sub_25061A0;
  v48 = sub_2507170;
  v47 = (void (__fastcall *)(__int64, _QWORD *, __int64))sub_25061D0;
  v45 = sub_2507080;
  v44 = (void (__fastcall *)(__int64, _QWORD *, __int64))sub_2506200;
  v14 = sub_A777F0(0x108u, a4);
  v15 = v14;
  if ( v14 )
  {
    *(_BYTE *)(v14 + 2) = 1;
    *(_WORD *)v14 = 257;
    *(_QWORD *)(v14 + 24) = 0;
    if ( v50 )
    {
      v50((__int64 *)(v14 + 8), v49, 2);
      *(_QWORD *)(v15 + 32) = v51;
      *(_QWORD *)(v15 + 24) = v50;
    }
    v16 = v47;
    *(_QWORD *)(v15 + 56) = 0;
    if ( v16 )
    {
      v16(v15 + 40, v46, 2);
      *(_QWORD *)(v15 + 64) = v48;
      *(_QWORD *)(v15 + 56) = v47;
    }
    v17 = v44;
    *(_QWORD *)(v15 + 88) = 0;
    if ( v17 )
    {
      v17(v15 + 72, v43, 2);
      *(_QWORD *)(v15 + 96) = v45;
      *(_QWORD *)(v15 + 88) = v44;
    }
    *(_QWORD *)(v15 + 104) = 0;
    *(_QWORD *)(v15 + 112) = 0;
    *(_QWORD *)(v15 + 120) = 0;
    *(_DWORD *)(v15 + 128) = 0;
    *(_QWORD *)(v15 + 136) = 0;
    *(_QWORD *)(v15 + 144) = 0;
    *(_QWORD *)(v15 + 152) = 0;
    *(_DWORD *)(v15 + 160) = 0;
    *(_QWORD *)(v15 + 168) = 0;
    *(_QWORD *)(v15 + 176) = 0;
    *(_QWORD *)(v15 + 184) = 0;
    *(_DWORD *)(v15 + 192) = 0;
    sub_3106C40(v15 + 200, v15, 0);
  }
  v18 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v44;
  *(_QWORD *)(a1 + 120) = v15;
  if ( v18 )
    v18(v43, v43, 3);
  if ( v47 )
    v47((__int64)v46, v46, 3);
  if ( v50 )
    v50(v49, v49, 3);
  *(_QWORD *)(a1 + 400) = a1 + 416;
  *(_QWORD *)(a1 + 408) = 0x800000000LL;
  sub_31229E0(a1 + 736);
  v19 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 912) = a1 + 928;
  *(_QWORD *)(a1 + 984) = v19;
  *(_QWORD *)(a1 + 992) = a1 + 1040;
  *(_QWORD *)(a1 + 1000) = a1 + 1048;
  *(_QWORD *)(a1 + 920) = 0x200000000LL;
  *(_WORD *)(a1 + 1020) = 512;
  *(_QWORD *)(a1 + 1040) = &unk_49DA100;
  *(_WORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 1152) = a1 + 1136;
  *(_QWORD *)(a1 + 1048) = &unk_49DA0B0;
  *(_QWORD *)(a1 + 1072) = 0x1000000000LL;
  *(_QWORD *)(a1 + 904) = a2;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_DWORD *)(a1 + 1016) = 0;
  *(_BYTE *)(a1 + 1022) = 7;
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_QWORD *)(a1 + 1056) = 0;
  *(_QWORD *)(a1 + 1064) = 0;
  *(_QWORD *)(a1 + 1080) = 0;
  *(_QWORD *)(a1 + 1088) = 0;
  *(_QWORD *)(a1 + 1096) = 0;
  *(_DWORD *)(a1 + 1104) = 0;
  *(_QWORD *)(a1 + 1112) = a1 + 400;
  *(_DWORD *)(a1 + 1120) = 0;
  *(_DWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1144) = 0;
  *(_QWORD *)(a1 + 1160) = a1 + 1136;
  *(_QWORD *)(a1 + 1200) = a1 + 1184;
  *(_QWORD *)(a1 + 1208) = a1 + 1184;
  *(_QWORD *)(a1 + 1240) = 0x6000000000LL;
  *(_QWORD *)(a1 + 1168) = 0;
  *(_DWORD *)(a1 + 1184) = 0;
  *(_QWORD *)(a1 + 1192) = 0;
  *(_QWORD *)(a1 + 1216) = 0;
  *(_QWORD *)(a1 + 1224) = 0;
  *(_QWORD *)(a1 + 1232) = 0;
  *(_QWORD *)(a1 + 1248) = a1 + 1264;
  sub_266F100((__int64 *)(a1 + 1248), *(_BYTE **)(a2 + 232), *(_QWORD *)(a2 + 232) + *(_QWORD *)(a2 + 240));
  *(_QWORD *)(a1 + 1280) = *(_QWORD *)(a2 + 264);
  *(_QWORD *)(a1 + 1288) = *(_QWORD *)(a2 + 272);
  v20 = *(_QWORD *)(a2 + 280);
  *(_QWORD *)(a1 + 1312) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2736) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2896) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2920) = a1 + 2936;
  *(_QWORD *)(a1 + 1296) = v20;
  *(_QWORD *)(a1 + 2928) = 0x400000000LL;
  *(_QWORD *)(a1 + 1304) = a1 + 1320;
  *(_QWORD *)(a1 + 2968) = a1 + 2984;
  *(_QWORD *)(a1 + 2728) = a1 + 2744;
  *(_QWORD *)(a1 + 2872) = 0;
  *(_QWORD *)(a1 + 2880) = 0;
  *(_QWORD *)(a1 + 2888) = 0;
  *(_QWORD *)(a1 + 2904) = 0;
  *(_QWORD *)(a1 + 2912) = 0;
  *(_QWORD *)(a1 + 2976) = 0;
  *(_QWORD *)(a1 + 2984) = 0;
  *(_QWORD *)(a1 + 2992) = 1;
  *(_QWORD *)(a1 + 3000) = 0;
  *(_QWORD *)(a1 + 3008) = 0;
  *(_QWORD *)(a1 + 3016) = 0;
  *(_QWORD *)(a1 + 3024) = 0;
  *(_QWORD *)(a1 + 3032) = 0;
  *(_QWORD *)(a1 + 3040) = 0;
  *(_QWORD *)(a1 + 3048) = 0;
  *(_QWORD *)(a1 + 3056) = 0;
  *(_QWORD *)(a1 + 3064) = 0;
  *(_QWORD *)(a1 + 3072) = 0;
  *(_QWORD *)(a1 + 3080) = 0;
  *(_QWORD *)(a1 + 3088) = 0;
  *(_QWORD *)(a1 + 3096) = 0;
  *(_QWORD *)(a1 + 3104) = 0;
  *(_QWORD *)(a1 + 3112) = 0;
  *(_QWORD *)(a1 + 3120) = 0;
  *(_QWORD *)(a1 + 3128) = 0;
  *(_QWORD *)(a1 + 3136) = 0;
  *(_QWORD *)(a1 + 3144) = 0;
  *(_QWORD *)(a1 + 3152) = 0;
  *(_QWORD *)(a1 + 3160) = 0;
  *(_QWORD *)(a1 + 3168) = 0;
  *(_QWORD *)(a1 + 3176) = 0;
  *(_QWORD *)(a1 + 3184) = 0;
  *(_QWORD *)(a1 + 3192) = 0;
  *(_QWORD *)(a1 + 3200) = 0;
  *(_QWORD *)(a1 + 3208) = 0;
  *(_QWORD *)(a1 + 3216) = 0;
  *(_QWORD *)(a1 + 3224) = 0;
  *(_QWORD *)(a1 + 3232) = 0;
  *(_QWORD *)(a1 + 3240) = 0;
  *(_QWORD *)(a1 + 3248) = 0;
  *(_QWORD *)(a1 + 3256) = 0;
  *(_QWORD *)(a1 + 3264) = 0;
  *(_QWORD *)(a1 + 3272) = 0;
  *(_QWORD *)(a1 + 3280) = 0;
  *(_QWORD *)(a1 + 3288) = 0;
  *(_QWORD *)(a1 + 3296) = 0;
  *(_QWORD *)(a1 + 3304) = 0;
  *(_QWORD *)(a1 + 3312) = 0;
  *(_QWORD *)(a1 + 3320) = 0;
  *(_QWORD *)(a1 + 3328) = 0;
  *(_QWORD *)(a1 + 3336) = 0;
  *(_QWORD *)(a1 + 3344) = 0;
  *(_QWORD *)(a1 + 3352) = 0;
  *(_QWORD *)(a1 + 3360) = 0;
  *(_QWORD *)(a1 + 3368) = 0;
  *(_QWORD *)(a1 + 3376) = 0;
  *(_QWORD *)(a1 + 3384) = 0;
  *(_QWORD *)(a1 + 3392) = 0;
  *(_QWORD *)(a1 + 3400) = 0;
  *(_QWORD *)(a1 + 3408) = 0;
  *(_QWORD *)(a1 + 3416) = 0;
  *(_QWORD *)(a1 + 3424) = 0;
  *(_QWORD *)(a1 + 3432) = 0;
  *(_QWORD *)(a1 + 3440) = 0;
  *(_QWORD *)(a1 + 3448) = 0;
  *(_QWORD *)(a1 + 3456) = 0;
  *(_QWORD *)(a1 + 3464) = 0;
  *(_QWORD *)(a1 + 3472) = 0;
  *(_QWORD *)(a1 + 3480) = a1 + 3496;
  v49[0] = 16;
  v21 = (_OWORD *)sub_22409D0(a1 + 3480, (unsigned __int64 *)v49, 0);
  v22 = v49[0];
  si128 = _mm_load_si128((const __m128i *)&xmmword_438FCF0);
  *(_QWORD *)(a1 + 3480) = v21;
  *(_QWORD *)(a1 + 3496) = v22;
  *v21 = si128;
  v24 = v49[0];
  v25 = *(_QWORD *)(a1 + 3480);
  *(_QWORD *)(a1 + 3488) = v49[0];
  *(_BYTE *)(v25 + v24) = 0;
  v26 = a1 + 3512;
  do
  {
    v27 = v26 + 56;
    *(_QWORD *)(v26 + 8) = 0;
    v26 += 160;
    *(_QWORD *)(v26 - 144) = 0;
    *(_QWORD *)(v26 - 120) = v27;
    *(_DWORD *)(v26 - 112) = 0;
    *(_DWORD *)(v26 - 108) = 8;
    *(_QWORD *)(v26 - 40) = 0;
    *(_QWORD *)(v26 - 32) = 0;
    *(_DWORD *)(v26 - 8) = 0;
    *(_QWORD *)(v26 - 24) = 0;
    *(_DWORD *)(v26 - 16) = 0;
    *(_DWORD *)(v26 - 12) = 0;
  }
  while ( a1 + 34552 != v26 );
  *(_QWORD *)(a1 + 34552) = 0;
  v28 = (_QWORD *)(a1 + 34592);
  *(_QWORD *)(a1 + 34560) = 0;
  *(_QWORD *)(a1 + 34568) = 0;
  *(_DWORD *)(a1 + 34576) = 0;
  do
  {
    *v28 = 0;
    v28 += 9;
    *(v28 - 8) = 0;
    *(v28 - 7) = 0;
    *(v28 - 6) = 0;
  }
  while ( (_QWORD *)(a1 + 34952) != v28 );
  *(_BYTE *)(a1 + 34976) = a6;
  v29 = *(_QWORD *)(a1 + 904);
  *(_QWORD *)(a1 + 34944) = 0;
  *(_QWORD *)(a1 + 34952) = 0;
  *(_QWORD *)(a1 + 34960) = 0;
  *(_DWORD *)(a1 + 34968) = 0;
  v30 = sub_2674830(v29);
  *(_BYTE *)(a1 + 737) = 1;
  *(_BYTE *)(a1 + 736) = v30;
  v31 = *(_QWORD *)(a1 + 904);
  v49[0] = (__int64)&v50;
  v42 = (_QWORD *)v31;
  sub_266F100(v49, *(_BYTE **)(v31 + 232), *(_QWORD *)(v31 + 232) + *(_QWORD *)(v31 + 240));
  v52 = v42[33];
  v53 = v42[34];
  v54 = v42[35];
  if ( (_DWORD)v52 == 27 || (unsigned int)(v52 - 42) <= 1 )
    *(_WORD *)(a1 + 738) = 257;
  else
    *(_WORD *)(a1 + 738) = 256;
  sub_3136900(a1 + 400);
  sub_2686D90(a1, a2, v32, v33, v34, v35);
  *(_QWORD *)(a1 + 34600) = 8;
  *(_QWORD *)(a1 + 34592) = "nthreads";
  *(_QWORD *)(a1 + 34608) = "OMP_NUM_THREADS";
  *(_QWORD *)(a1 + 34664) = "active_levels";
  *(_QWORD *)(a1 + 34680) = "NONE";
  v36 = *(_QWORD ***)(a1 + 3032);
  *(_DWORD *)(a1 + 34584) = 0;
  *(_DWORD *)(a1 + 34624) = 2;
  *(_QWORD *)(a1 + 34616) = 15;
  *(_QWORD *)(a1 + 34632) = 0;
  *(_QWORD *)(a1 + 34672) = 13;
  *(_DWORD *)(a1 + 34656) = 1;
  *(_DWORD *)(a1 + 34696) = 0;
  *(_QWORD *)(a1 + 34688) = 4;
  v37 = sub_BCB2D0(*v36);
  v38 = sub_ACD640(v37, 0, 0);
  *(_QWORD *)(a1 + 34744) = 6;
  *(_QWORD *)(a1 + 34704) = v38;
  *(_QWORD *)(a1 + 34736) = "cancel";
  *(_QWORD *)(a1 + 34752) = "OMP_CANCELLATION";
  v39 = *(__int64 ***)(a1 + 3008);
  *(_DWORD *)(a1 + 34728) = 2;
  *(_DWORD *)(a1 + 34768) = 1;
  *(_QWORD *)(a1 + 34760) = 16;
  v40 = sub_ACD720(*v39);
  v41 = (__int64 (__fastcall **)(__int64 *, __int64 *, int))v49[0];
  *(_QWORD *)(a1 + 34816) = 9;
  *(_QWORD *)(a1 + 34776) = v40;
  *(_QWORD *)(a1 + 34808) = "proc_bind";
  *(_QWORD *)(a1 + 34824) = "OMP_PROC_BIND";
  *(_QWORD *)(a1 + 34880) = "__last";
  *(_QWORD *)(a1 + 34896) = "last";
  *(_DWORD *)(a1 + 34800) = 3;
  *(_DWORD *)(a1 + 34840) = 2;
  *(_QWORD *)(a1 + 34832) = 13;
  *(_QWORD *)(a1 + 34848) = 0;
  *(_QWORD *)(a1 + 34888) = 6;
  *(_DWORD *)(a1 + 34872) = 4;
  *(_DWORD *)(a1 + 34912) = 3;
  *(_QWORD *)(a1 + 34904) = 4;
  *(_QWORD *)(a1 + 34640) = 0x1300000029LL;
  *(_DWORD *)(a1 + 34716) = 31;
  *(_DWORD *)(a1 + 34788) = 22;
  *(_DWORD *)(a1 + 34860) = 33;
  if ( v41 != &v50 )
    j_j___libc_free_0((unsigned __int64)v41);
}
