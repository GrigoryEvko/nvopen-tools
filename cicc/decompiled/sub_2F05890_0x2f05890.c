// Function: sub_2F05890
// Address: 0x2f05890
//
void __fastcall sub_2F05890(__int64 a1, const char *a2)
{
  void *v2; // rax
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD v8[10]; // [rsp+0h] [rbp-340h] BYREF
  __int16 v9; // [rsp+50h] [rbp-2F0h]
  char v10; // [rsp+52h] [rbp-2EEh]
  __int64 v11; // [rsp+58h] [rbp-2E8h]
  __int64 v12; // [rsp+60h] [rbp-2E0h]
  __int64 v13; // [rsp+68h] [rbp-2D8h]
  char *v14; // [rsp+70h] [rbp-2D0h]
  __int64 v15; // [rsp+78h] [rbp-2C8h]
  int v16; // [rsp+80h] [rbp-2C0h]
  char v17; // [rsp+84h] [rbp-2BCh]
  char v18; // [rsp+88h] [rbp-2B8h] BYREF
  char *v19; // [rsp+C8h] [rbp-278h]
  __int64 v20; // [rsp+D0h] [rbp-270h]
  char v21; // [rsp+D8h] [rbp-268h] BYREF
  int v22; // [rsp+108h] [rbp-238h]
  __int64 v23; // [rsp+110h] [rbp-230h]
  __int64 v24; // [rsp+118h] [rbp-228h]
  __int64 v25; // [rsp+120h] [rbp-220h]
  __int64 v26; // [rsp+128h] [rbp-218h]
  char *v27; // [rsp+130h] [rbp-210h]
  __int64 v28; // [rsp+138h] [rbp-208h]
  char v29; // [rsp+140h] [rbp-200h] BYREF
  char *v30; // [rsp+180h] [rbp-1C0h]
  __int64 v31; // [rsp+188h] [rbp-1B8h]
  char v32; // [rsp+190h] [rbp-1B0h] BYREF
  char *v33; // [rsp+1D0h] [rbp-170h]
  __int64 v34; // [rsp+1D8h] [rbp-168h]
  char v35; // [rsp+1E0h] [rbp-160h] BYREF
  char *v36; // [rsp+220h] [rbp-120h]
  __int64 v37; // [rsp+228h] [rbp-118h]
  char v38; // [rsp+230h] [rbp-110h] BYREF
  __int64 v39; // [rsp+250h] [rbp-F0h]
  __int64 v40; // [rsp+258h] [rbp-E8h]
  __int64 v41; // [rsp+260h] [rbp-E0h]
  __int64 v42; // [rsp+268h] [rbp-D8h]
  int v43; // [rsp+270h] [rbp-D0h]
  __int64 v44; // [rsp+278h] [rbp-C8h]
  __int64 v45; // [rsp+280h] [rbp-C0h]
  __int64 v46; // [rsp+288h] [rbp-B8h]
  __int64 v47; // [rsp+290h] [rbp-B0h]
  int v48; // [rsp+298h] [rbp-A8h]
  char v49; // [rsp+29Ch] [rbp-A4h]
  char *v50; // [rsp+2A0h] [rbp-A0h]
  __int64 v51; // [rsp+2A8h] [rbp-98h]
  char v52; // [rsp+2B0h] [rbp-90h] BYREF
  char *v53; // [rsp+2B8h] [rbp-88h]
  __int64 v54; // [rsp+2C0h] [rbp-80h]
  char v55; // [rsp+2C8h] [rbp-78h] BYREF
  __int64 v56; // [rsp+300h] [rbp-40h]
  __int64 v57; // [rsp+308h] [rbp-38h]
  char v58; // [rsp+310h] [rbp-30h]
  __int64 v59; // [rsp+314h] [rbp-2Ch]

  v2 = sub_CB72A0();
  v8[1] = a1;
  v3 = *(_QWORD *)(a1 + 200);
  v8[0] = 0;
  if ( !v2 )
    v2 = sub_CB7330();
  v8[2] = v2;
  v9 = 0;
  v14 = &v18;
  v19 = &v21;
  v27 = &v29;
  v28 = 0x1000000000LL;
  v31 = 0x1000000000LL;
  v34 = 0x1000000000LL;
  v36 = &v38;
  v30 = &v32;
  v37 = 0x400000000LL;
  v20 = 0x600000000LL;
  v33 = &v35;
  v8[3] = v3;
  memset(&v8[4], 0, 48);
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v15 = 8;
  v16 = 0;
  v17 = 1;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v39 = 0;
  v50 = &v52;
  v51 = 0x100000000LL;
  v54 = 0x600000000LL;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 1;
  v53 = &v55;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  sub_2F02A10((__int64)v8, a2);
  sub_2EF2DE0((__int64)v8, (__int64)a2, v4, v5, v6, v7);
}
