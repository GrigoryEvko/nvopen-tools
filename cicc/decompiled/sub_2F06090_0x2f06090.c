// Function: sub_2F06090
// Address: 0x2f06090
//
__int64 __fastcall sub_2F06090(const char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, char a6)
{
  unsigned int v9; // eax
  unsigned int v10; // r12d
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD v16[10]; // [rsp+0h] [rbp-350h] BYREF
  __int16 v17; // [rsp+50h] [rbp-300h]
  char v18; // [rsp+52h] [rbp-2FEh]
  __int64 v19; // [rsp+58h] [rbp-2F8h]
  __int64 v20; // [rsp+60h] [rbp-2F0h]
  __int64 v21; // [rsp+68h] [rbp-2E8h]
  char *v22; // [rsp+70h] [rbp-2E0h]
  __int64 v23; // [rsp+78h] [rbp-2D8h]
  int v24; // [rsp+80h] [rbp-2D0h]
  char v25; // [rsp+84h] [rbp-2CCh]
  char v26; // [rsp+88h] [rbp-2C8h] BYREF
  char *v27; // [rsp+C8h] [rbp-288h]
  __int64 v28; // [rsp+D0h] [rbp-280h]
  char v29; // [rsp+D8h] [rbp-278h] BYREF
  int v30; // [rsp+108h] [rbp-248h]
  __int64 v31; // [rsp+110h] [rbp-240h]
  __int64 v32; // [rsp+118h] [rbp-238h]
  __int64 v33; // [rsp+120h] [rbp-230h]
  __int64 v34; // [rsp+128h] [rbp-228h]
  char *v35; // [rsp+130h] [rbp-220h]
  __int64 v36; // [rsp+138h] [rbp-218h]
  char v37; // [rsp+140h] [rbp-210h] BYREF
  char *v38; // [rsp+180h] [rbp-1D0h]
  __int64 v39; // [rsp+188h] [rbp-1C8h]
  char v40; // [rsp+190h] [rbp-1C0h] BYREF
  char *v41; // [rsp+1D0h] [rbp-180h]
  __int64 v42; // [rsp+1D8h] [rbp-178h]
  char v43; // [rsp+1E0h] [rbp-170h] BYREF
  char *v44; // [rsp+220h] [rbp-130h]
  __int64 v45; // [rsp+228h] [rbp-128h]
  char v46; // [rsp+230h] [rbp-120h] BYREF
  __int64 v47; // [rsp+250h] [rbp-100h]
  __int64 v48; // [rsp+258h] [rbp-F8h]
  __int64 v49; // [rsp+260h] [rbp-F0h]
  __int64 v50; // [rsp+268h] [rbp-E8h]
  int v51; // [rsp+270h] [rbp-E0h]
  __int64 v52; // [rsp+278h] [rbp-D8h]
  __int64 v53; // [rsp+280h] [rbp-D0h]
  __int64 v54; // [rsp+288h] [rbp-C8h]
  __int64 v55; // [rsp+290h] [rbp-C0h]
  int v56; // [rsp+298h] [rbp-B8h]
  char v57; // [rsp+29Ch] [rbp-B4h]
  char *v58; // [rsp+2A0h] [rbp-B0h]
  __int64 v59; // [rsp+2A8h] [rbp-A8h]
  char v60; // [rsp+2B0h] [rbp-A0h] BYREF
  char *v61; // [rsp+2B8h] [rbp-98h]
  __int64 v62; // [rsp+2C0h] [rbp-90h]
  char v63; // [rsp+2C8h] [rbp-88h] BYREF
  __int64 v64; // [rsp+300h] [rbp-50h]
  __int64 v65; // [rsp+308h] [rbp-48h]
  char v66; // [rsp+310h] [rbp-40h]
  __int64 v67; // [rsp+314h] [rbp-3Ch]

  v16[0] = 0;
  v16[1] = 0;
  if ( !a5 )
    a5 = sub_CB7330();
  v16[2] = a5;
  v17 = 0;
  v22 = &v26;
  v27 = &v29;
  v35 = &v37;
  v36 = 0x1000000000LL;
  v39 = 0x1000000000LL;
  v42 = 0x1000000000LL;
  v44 = &v46;
  v38 = &v40;
  v45 = 0x400000000LL;
  v28 = 0x600000000LL;
  v41 = &v43;
  v16[3] = a4;
  memset(&v16[4], 0, 48);
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v23 = 8;
  v24 = 0;
  v25 = 1;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v47 = 0;
  v58 = &v60;
  v59 = 0x100000000LL;
  v53 = a2;
  v55 = a3;
  v57 = a6;
  v62 = 0x600000000LL;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v54 = 0;
  v56 = 0;
  v61 = &v63;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  LOBYTE(v9) = sub_2F02A10((__int64)v16, a1);
  v10 = v9;
  sub_2EF2DE0((__int64)v16, (__int64)a1, v11, v12, v13, v14);
  return v10;
}
