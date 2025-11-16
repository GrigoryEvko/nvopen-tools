// Function: sub_2F05B50
// Address: 0x2f05b50
//
__int64 __fastcall sub_2F05B50(const char *a1, __int64 a2, __int64 a3, __int64 *a4, char a5)
{
  unsigned int v7; // eax
  unsigned int v8; // r12d
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD v14[10]; // [rsp+0h] [rbp-340h] BYREF
  __int16 v15; // [rsp+50h] [rbp-2F0h]
  char v16; // [rsp+52h] [rbp-2EEh]
  __int64 v17; // [rsp+58h] [rbp-2E8h]
  __int64 v18; // [rsp+60h] [rbp-2E0h]
  __int64 v19; // [rsp+68h] [rbp-2D8h]
  char *v20; // [rsp+70h] [rbp-2D0h]
  __int64 v21; // [rsp+78h] [rbp-2C8h]
  int v22; // [rsp+80h] [rbp-2C0h]
  char v23; // [rsp+84h] [rbp-2BCh]
  char v24; // [rsp+88h] [rbp-2B8h] BYREF
  char *v25; // [rsp+C8h] [rbp-278h]
  __int64 v26; // [rsp+D0h] [rbp-270h]
  char v27; // [rsp+D8h] [rbp-268h] BYREF
  int v28; // [rsp+108h] [rbp-238h]
  __int64 v29; // [rsp+110h] [rbp-230h]
  __int64 v30; // [rsp+118h] [rbp-228h]
  __int64 v31; // [rsp+120h] [rbp-220h]
  __int64 v32; // [rsp+128h] [rbp-218h]
  char *v33; // [rsp+130h] [rbp-210h]
  __int64 v34; // [rsp+138h] [rbp-208h]
  char v35; // [rsp+140h] [rbp-200h] BYREF
  char *v36; // [rsp+180h] [rbp-1C0h]
  __int64 v37; // [rsp+188h] [rbp-1B8h]
  char v38; // [rsp+190h] [rbp-1B0h] BYREF
  char *v39; // [rsp+1D0h] [rbp-170h]
  __int64 v40; // [rsp+1D8h] [rbp-168h]
  char v41; // [rsp+1E0h] [rbp-160h] BYREF
  char *v42; // [rsp+220h] [rbp-120h]
  __int64 v43; // [rsp+228h] [rbp-118h]
  char v44; // [rsp+230h] [rbp-110h] BYREF
  __int64 v45; // [rsp+250h] [rbp-F0h]
  __int64 v46; // [rsp+258h] [rbp-E8h]
  __int64 v47; // [rsp+260h] [rbp-E0h]
  __int64 v48; // [rsp+268h] [rbp-D8h]
  int v49; // [rsp+270h] [rbp-D0h]
  __int64 v50; // [rsp+278h] [rbp-C8h]
  __int64 v51; // [rsp+280h] [rbp-C0h]
  __int64 v52; // [rsp+288h] [rbp-B8h]
  __int64 v53; // [rsp+290h] [rbp-B0h]
  int v54; // [rsp+298h] [rbp-A8h]
  char v55; // [rsp+29Ch] [rbp-A4h]
  char *v56; // [rsp+2A0h] [rbp-A0h]
  __int64 v57; // [rsp+2A8h] [rbp-98h]
  char v58; // [rsp+2B0h] [rbp-90h] BYREF
  char *v59; // [rsp+2B8h] [rbp-88h]
  __int64 v60; // [rsp+2C0h] [rbp-80h]
  char v61; // [rsp+2C8h] [rbp-78h] BYREF
  __int64 v62; // [rsp+300h] [rbp-40h]
  __int64 v63; // [rsp+308h] [rbp-38h]
  char v64; // [rsp+310h] [rbp-30h]
  __int64 v65; // [rsp+314h] [rbp-2Ch]

  v14[1] = a2;
  v14[0] = 0;
  if ( !a4 )
    a4 = sub_CB7330();
  v14[2] = a4;
  v15 = 0;
  v20 = &v24;
  v25 = &v27;
  v33 = &v35;
  v34 = 0x1000000000LL;
  v37 = 0x1000000000LL;
  v40 = 0x1000000000LL;
  v42 = &v44;
  v36 = &v38;
  v43 = 0x400000000LL;
  v26 = 0x600000000LL;
  v39 = &v41;
  v14[3] = a3;
  memset(&v14[4], 0, 48);
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v21 = 8;
  v22 = 0;
  v23 = 1;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v45 = 0;
  v56 = &v58;
  v57 = 0x100000000LL;
  v55 = a5;
  v60 = 0x600000000LL;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v59 = &v61;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  LOBYTE(v7) = sub_2F02A10((__int64)v14, a1);
  v8 = v7;
  sub_2EF2DE0((__int64)v14, (__int64)a1, v9, v10, v11, v12);
  return v8;
}
