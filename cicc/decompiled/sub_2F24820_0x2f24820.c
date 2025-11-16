// Function: sub_2F24820
// Address: 0x2f24820
//
void __fastcall sub_2F24820(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  const char *v4; // rax
  __int64 v5; // rdx
  char v6; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i *v10; // rbx
  __m128i *v11; // r14
  __m128i *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 (*v16)(); // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  void (*v19)(void); // rax
  __int64 v20; // r15
  unsigned __int64 *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // [rsp+10h] [rbp-470h]
  __int64 *v27; // [rsp+28h] [rbp-458h]
  __int64 *v28; // [rsp+30h] [rbp-450h]
  _QWORD v29[3]; // [rsp+40h] [rbp-440h] BYREF
  _BYTE *v30; // [rsp+58h] [rbp-428h]
  _BYTE *v31; // [rsp+60h] [rbp-420h]
  __int64 v32; // [rsp+68h] [rbp-418h]
  _QWORD *v33; // [rsp+70h] [rbp-410h]
  _BYTE v34[144]; // [rsp+80h] [rbp-400h] BYREF
  __m128i v35; // [rsp+110h] [rbp-370h] BYREF
  __int64 *v36; // [rsp+120h] [rbp-360h]
  __int64 *v37; // [rsp+128h] [rbp-358h]
  _BYTE *v38; // [rsp+130h] [rbp-350h]
  __int64 v39; // [rsp+138h] [rbp-348h]
  _BYTE v40[128]; // [rsp+140h] [rbp-340h] BYREF
  const char *v41; // [rsp+1C0h] [rbp-2C0h] BYREF
  __int64 v42; // [rsp+1C8h] [rbp-2B8h]
  __int64 v43; // [rsp+1D0h] [rbp-2B0h]
  __int64 v44; // [rsp+1D8h] [rbp-2A8h]
  __int64 v45; // [rsp+1E0h] [rbp-2A0h]
  __int16 v46; // [rsp+1E8h] [rbp-298h]
  __int64 v47; // [rsp+1F0h] [rbp-290h]
  __int64 v48; // [rsp+1F8h] [rbp-288h]
  __int64 v49; // [rsp+200h] [rbp-280h]
  __int64 v50; // [rsp+208h] [rbp-278h]
  __int64 v51; // [rsp+210h] [rbp-270h]
  __int64 v52; // [rsp+218h] [rbp-268h]
  char v53; // [rsp+238h] [rbp-248h]
  int v54; // [rsp+240h] [rbp-240h] BYREF
  __int64 v55; // [rsp+248h] [rbp-238h]
  __int64 v56; // [rsp+250h] [rbp-230h]
  __int16 v57; // [rsp+258h] [rbp-228h]
  char *v58; // [rsp+260h] [rbp-220h]
  __int64 v59; // [rsp+268h] [rbp-218h]
  char v60; // [rsp+270h] [rbp-210h] BYREF
  __int64 v61; // [rsp+280h] [rbp-200h]
  __int64 v62; // [rsp+288h] [rbp-1F8h]
  char *v63; // [rsp+290h] [rbp-1F0h]
  __int64 v64; // [rsp+298h] [rbp-1E8h]
  char v65; // [rsp+2A0h] [rbp-1E0h] BYREF
  __int64 v66; // [rsp+2B0h] [rbp-1D0h]
  __int64 v67; // [rsp+2B8h] [rbp-1C8h]
  __int64 v68; // [rsp+2C0h] [rbp-1C0h]
  int v69; // [rsp+2C8h] [rbp-1B8h]
  char v70; // [rsp+2CCh] [rbp-1B4h]
  int v71; // [rsp+2D0h] [rbp-1B0h]
  char *v72; // [rsp+2D8h] [rbp-1A8h]
  __int64 v73; // [rsp+2E0h] [rbp-1A0h]
  char v74; // [rsp+2E8h] [rbp-198h] BYREF
  __int64 v75; // [rsp+2F8h] [rbp-188h]
  __int64 v76; // [rsp+300h] [rbp-180h]
  char *v77; // [rsp+308h] [rbp-178h]
  __int64 v78; // [rsp+310h] [rbp-170h]
  char v79; // [rsp+318h] [rbp-168h] BYREF
  __int64 v80; // [rsp+328h] [rbp-158h]
  __int64 v81; // [rsp+330h] [rbp-150h]
  __int64 v82; // [rsp+338h] [rbp-148h]
  __int64 v83; // [rsp+340h] [rbp-140h]
  __int64 v84; // [rsp+348h] [rbp-138h]
  __int64 v85; // [rsp+350h] [rbp-130h]
  __int64 v86; // [rsp+358h] [rbp-128h]
  __int64 v87; // [rsp+360h] [rbp-120h]
  __int64 v88; // [rsp+368h] [rbp-118h]
  __int64 v89; // [rsp+370h] [rbp-110h]
  __int64 v90; // [rsp+378h] [rbp-108h]
  __int64 v91; // [rsp+380h] [rbp-100h]
  __int64 v92; // [rsp+388h] [rbp-F8h]
  __int64 v93; // [rsp+390h] [rbp-F0h]
  unsigned __int64 v94; // [rsp+398h] [rbp-E8h]
  __int64 v95; // [rsp+3A0h] [rbp-E0h]
  __int64 v96; // [rsp+3A8h] [rbp-D8h]
  __int64 v97; // [rsp+3B0h] [rbp-D0h]
  __int64 v98; // [rsp+3B8h] [rbp-C8h] BYREF
  __m128i *v99; // [rsp+3C0h] [rbp-C0h]
  __m128i *v100; // [rsp+3C8h] [rbp-B8h]
  int v101; // [rsp+3D0h] [rbp-B0h] BYREF
  __int64 v102; // [rsp+3D8h] [rbp-A8h]
  __int64 v103; // [rsp+3E0h] [rbp-A0h]
  __int64 v104; // [rsp+3E8h] [rbp-98h]
  __int64 v105; // [rsp+3F0h] [rbp-90h]
  __int64 v106; // [rsp+3F8h] [rbp-88h]
  __int64 v107; // [rsp+400h] [rbp-80h]
  __int64 v108; // [rsp+408h] [rbp-78h]
  __int64 v109; // [rsp+410h] [rbp-70h]
  __int64 v110; // [rsp+418h] [rbp-68h]
  _QWORD v111[2]; // [rsp+420h] [rbp-60h] BYREF
  char v112; // [rsp+430h] [rbp-50h] BYREF
  __int64 v113; // [rsp+440h] [rbp-40h]
  __int64 v114; // [rsp+448h] [rbp-38h]

  v2 = (__int64 *)a1;
  sub_2F1A3B0(a1, a2);
  v46 = 0;
  v58 = &v60;
  v63 = &v65;
  v68 = 0xFFFFFFFFLL;
  v57 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = &v74;
  v77 = &v79;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 6;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111[0] = &v112;
  v111[1] = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v4 = sub_2E791E0((__int64 *)a2);
  BYTE1(v43) = 1;
  v41 = v4;
  LOBYTE(v4) = *(_BYTE *)(a2 + 340);
  v42 = v5;
  LODWORD(v5) = *(_DWORD *)(a2 + 576);
  LOBYTE(v43) = (_BYTE)v4;
  LOBYTE(v4) = *(_BYTE *)(a2 + 341);
  *(_DWORD *)((char *)&v45 + 1) = v5;
  BYTE2(v43) = (_BYTE)v4;
  LOBYTE(v4) = *(_BYTE *)(a2 + 343);
  LOBYTE(v45) = 1;
  LOBYTE(v44) = (_BYTE)v4;
  BYTE5(v45) = *(_BYTE *)(a2 + 580);
  HIBYTE(v44) = *(_BYTE *)(a2 + 581);
  BYTE6(v45) = *(_BYTE *)(a2 + 582);
  v6 = sub_2E799E0(a2);
  v7 = *(_QWORD *)(a2 + 16);
  BYTE2(v44) = 1;
  HIBYTE(v46) = v6;
  v8 = *(_QWORD *)(a2 + 344);
  BYTE4(v44) = 1;
  BYTE6(v44) = 1;
  BYTE3(v43) = (v8 & 0x20) != 0;
  BYTE4(v43) = (v8 & 0x40) != 0;
  BYTE5(v43) = (unsigned __int8)v8 >> 7;
  BYTE6(v43) = (v8 & 0x10) != 0;
  HIBYTE(v45) = (v8 & 0x200) != 0;
  LOBYTE(v46) = (v8 & 0x800) != 0;
  BYTE1(v44) = (v8 & 2) != 0;
  BYTE3(v44) = v8 & 1;
  BYTE5(v44) = (v8 & 8) != 0;
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 200LL))(v7);
  sub_2F151D0((__int64)v2, (__int64)&v41, a2, *(_QWORD *)(a2 + 32), v9);
  sub_3531F50(v34, v2[1], a2, 1);
  sub_A564B0((__int64)v34, *(_QWORD *)a2);
  sub_2F10960((__int64)v2, (__int64)v34, (__int64)&v54, *(_QWORD *)(a2 + 48));
  sub_2F16670((__int64)v2, &v41, a2, (__int64)v34);
  sub_2F13450((__int64)v2, (unsigned __int64 *)&v41, a2, (__int64)v34);
  sub_2F232C0((__int64)v2, &v41, a2);
  v10 = *(__m128i **)(a2 + 904);
  if ( v10 != (__m128i *)((char *)v10 + 20 * *(unsigned int *)(a2 + 912)) )
  {
    v11 = (__m128i *)((char *)v10 + 20 * *(unsigned int *)(a2 + 912));
    do
    {
      while ( 1 )
      {
        v12 = v99;
        v35 = *v10;
        LODWORD(v36) = v10[1].m128i_i32[0];
        if ( v99 != v100 )
          break;
        v10 = (__m128i *)((char *)v10 + 20);
        sub_2F14360((__int64)&v98, v99, &v35);
        if ( v11 == v10 )
          goto LABEL_8;
      }
      if ( v99 )
      {
        *v99 = _mm_loadu_si128(&v35);
        v12[1].m128i_i32[0] = (int)v36;
        v12 = v99;
      }
      v10 = (__m128i *)((char *)v10 + 20);
      v99 = (__m128i *)((char *)v12 + 20);
    }
    while ( v11 != v10 );
  }
LABEL_8:
  v13 = *(_QWORD *)(a2 + 56);
  if ( v13 )
    sub_2F19560((__int64)v2, (unsigned __int64 *)&v41, v13);
  v14 = *(_QWORD *)(a2 + 64);
  if ( v14 )
    sub_2F19D40((__int64)v2, (__int64)v34, (__int64)&v101, v14);
  v15 = *(_QWORD *)(a2 + 8);
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 64LL);
  v17 = 0;
  if ( v16 != sub_23CE2D0 )
    v17 = ((__int64 (__fastcall *)(__int64, __int64))v16)(v15, a2);
  v18 = v94;
  v94 = v17;
  if ( v18 )
  {
    v19 = *(void (**)(void))(*(_QWORD *)v18 + 8LL);
    if ( (char *)v19 == (char *)sub_2F07240 )
      j_j___libc_free_0(v18);
    else
      v19();
  }
  v32 = 0x100000000LL;
  v29[1] = 0;
  v29[2] = 0;
  v29[0] = &unk_49DD210;
  v33 = v111;
  v30 = 0;
  v31 = 0;
  sub_CB5980((__int64)v29, 0, 0, 0);
  v28 = v2 + 6;
  v27 = v2 + 2;
  if ( a2 + 320 != *(_QWORD *)(a2 + 328) )
  {
    v26 = v2;
    v20 = *(_QWORD *)(a2 + 328);
    while ( 1 )
    {
      v35.m128i_i64[0] = (__int64)v29;
      v35.m128i_i64[1] = (__int64)v34;
      v36 = v27;
      v38 = v40;
      v37 = v28;
      v39 = 0x800000000LL;
      sub_2F12790(v35.m128i_i64, v20);
      if ( v38 != v40 )
        _libc_free((unsigned __int64)v38);
      v20 = *(_QWORD *)(v20 + 8);
      if ( a2 + 320 == v20 )
        break;
      if ( v30 == v31 )
        sub_CB6200((__int64)v29, (unsigned __int8 *)"\n", 1u);
      else
        *v31++ = 10;
    }
    v2 = v26;
  }
  sub_2F18B70((__int64)v2, (unsigned __int64 *)&v41, a2, (__int64)v34);
  sub_2F244E0((__int64)v2, &v41, a2);
  sub_CB1A80((__int64)&v35, *v2, 0, 70);
  if ( !(_BYTE)qword_50225C8 )
    v40[47] = 1;
  sub_CB2850((__int64)&v35);
  v21 = 0;
  if ( (unsigned __int8)sub_CB2870((__int64)&v35, 0) )
  {
    sub_CB05C0(&v35, 0, v22, v23, v24, v25);
    v21 = (unsigned __int64 *)&v41;
    sub_2F1FBF0((__int64)&v35, (__int64)&v41);
    sub_CB2220(&v35);
    nullsub_173();
  }
  sub_CB1B70((__int64)&v35);
  sub_CB0A00(&v35, (__int64)v21);
  v29[0] = &unk_49DD210;
  sub_CB5840((__int64)v29);
  sub_3531BA0(v34);
  sub_2F101D0((__int64)&v41);
}
