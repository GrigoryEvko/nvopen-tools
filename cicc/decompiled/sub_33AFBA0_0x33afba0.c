// Function: sub_33AFBA0
// Address: 0x33afba0
//
void __fastcall sub_33AFBA0(__int64 a1, __int64 a2)
{
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __m128i v20; // rax
  int v21; // r9d
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned int v33; // edx
  unsigned __int16 *v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // r8
  __int64 *v42; // rdx
  __int64 v43; // rax
  unsigned int v44; // edx
  unsigned __int16 *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // r8
  __int64 *v53; // rdx
  int v54; // eax
  int v55; // edx
  int v56; // r9d
  __int64 v57; // rax
  __int64 v58; // r14
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // r13
  __m128i v62; // rax
  __m128i v63; // rax
  int v64; // eax
  int v65; // edx
  int v66; // r9d
  __m128i v67; // xmm0
  __m128i v68; // xmm1
  __int64 v69; // rsi
  __int64 v70; // r14
  __int64 v71; // rdx
  __int64 v72; // r13
  __int64 v73; // r15
  __int64 v74; // rsi
  __int128 v75; // [rsp-10h] [rbp-330h]
  __int128 v76; // [rsp-10h] [rbp-330h]
  __int128 v77; // [rsp-10h] [rbp-330h]
  __int64 v78; // [rsp-8h] [rbp-328h]
  __m128i v79; // [rsp+0h] [rbp-320h] BYREF
  __m128i v80; // [rsp+10h] [rbp-310h] BYREF
  unsigned __int64 v81; // [rsp+20h] [rbp-300h]
  _BYTE *v82; // [rsp+28h] [rbp-2F8h]
  __int64 v83; // [rsp+30h] [rbp-2F0h]
  __int64 v84; // [rsp+38h] [rbp-2E8h]
  __int64 v85; // [rsp+40h] [rbp-2E0h]
  __int64 v86; // [rsp+48h] [rbp-2D8h]
  __int64 v87; // [rsp+50h] [rbp-2D0h]
  __int64 v88; // [rsp+58h] [rbp-2C8h]
  __int64 v89; // [rsp+60h] [rbp-2C0h]
  __int64 v90; // [rsp+68h] [rbp-2B8h]
  __int64 v91; // [rsp+70h] [rbp-2B0h]
  __int64 v92; // [rsp+78h] [rbp-2A8h]
  __int64 v93; // [rsp+80h] [rbp-2A0h] BYREF
  int v94; // [rsp+88h] [rbp-298h]
  _QWORD *v95; // [rsp+90h] [rbp-290h] BYREF
  __int64 v96; // [rsp+98h] [rbp-288h]
  __int64 v97; // [rsp+A0h] [rbp-280h] BYREF
  __int64 v98; // [rsp+A8h] [rbp-278h]
  __m128i v99; // [rsp+B0h] [rbp-270h]
  __m128i v100; // [rsp+C0h] [rbp-260h]
  __int64 v101; // [rsp+D0h] [rbp-250h]
  __int64 v102; // [rsp+D8h] [rbp-248h]
  _BYTE *v103; // [rsp+E0h] [rbp-240h] BYREF
  __int64 v104; // [rsp+E8h] [rbp-238h]
  _BYTE v105[560]; // [rsp+F0h] [rbp-230h] BYREF

  v82 = v105;
  v4 = *(_DWORD *)(a1 + 848);
  v103 = v105;
  v104 = 0x2000000000LL;
  v5 = *(_QWORD *)a1;
  v93 = 0;
  v94 = v4;
  if ( v5 )
  {
    if ( &v93 != (__int64 *)(v5 + 48) )
    {
      v6 = *(_QWORD *)(v5 + 48);
      v93 = v6;
      if ( v6 )
        sub_B96E90((__int64)&v93, v6, 1);
    }
  }
  v7 = *(_QWORD *)(a2 - 32);
  sub_338B750(a1, v7);
  v8 = *(_QWORD *)(a1 + 864);
  v13 = sub_33738B0(a1, v7, v9, v10, v11, v12);
  v79.m128i_i64[0] = v14;
  v81 = v13;
  v15 = sub_33E5110(v8, 1, 0, 262, 0);
  v80.m128i_i64[0] = v16;
  v17 = v15;
  v95 = (_QWORD *)v81;
  LODWORD(v96) = v79.m128i_i32[0];
  v18 = sub_3400D50(v8, 0, &v93, 1);
  v98 = v19;
  v97 = v18;
  v20.m128i_i64[0] = sub_3400D50(v8, 0, &v93, 1);
  v99 = v20;
  *((_QWORD *)&v75 + 1) = 3;
  *(_QWORD *)&v75 = &v95;
  v91 = sub_3411630(v8, 315, (unsigned int)&v93, v17, v80.m128i_i32[0], v21, v75);
  v22 = v91;
  v24 = (unsigned int)v23;
  v25 = v78;
  v26 = (unsigned int)v104;
  v92 = v23;
  v27 = (unsigned int)v104 + 1LL;
  if ( v27 > HIDWORD(v104) )
  {
    v80.m128i_i64[0] = v24;
    sub_C8D5F0((__int64)&v103, v82, v27, 0x10u, v24, v78);
    v26 = (unsigned int)v104;
    v24 = v80.m128i_i64[0];
  }
  v28 = (__int64 *)&v103[16 * v26];
  *v28 = v91;
  v28[1] = v24;
  LODWORD(v104) = v104 + 1;
  v29 = (unsigned int)v104;
  if ( (unsigned __int64)(unsigned int)v104 + 1 > HIDWORD(v104) )
  {
    sub_C8D5F0((__int64)&v103, v82, (unsigned int)v104 + 1LL, 0x10u, v24, v25);
    v29 = (unsigned int)v104;
  }
  v30 = (__int64 *)&v103[16 * v29];
  *v30 = v22;
  v30[1] = 1;
  LODWORD(v30) = *(_DWORD *)(a2 + 4);
  LODWORD(v104) = v104 + 1;
  v31 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * ((unsigned int)v30 & 0x7FFFFFF)));
  v32 = *(_QWORD *)(a1 + 864);
  v34 = (unsigned __int16 *)(*(_QWORD *)(v31 + 48) + 16LL * v33);
  v35 = *(_QWORD *)(v31 + 96);
  if ( *(_DWORD *)(v35 + 32) <= 0x40u )
    v36 = *(_QWORD *)(v35 + 24);
  else
    v36 = **(_QWORD **)(v35 + 24);
  v37 = sub_3400BD0(v32, v36, (unsigned int)&v93, *v34, *((_QWORD *)v34 + 1), 1, 0);
  v39 = v38;
  v40 = (unsigned int)v104;
  v41 = v37;
  if ( (unsigned __int64)(unsigned int)v104 + 1 > HIDWORD(v104) )
  {
    v80.m128i_i64[0] = v37;
    v80.m128i_i64[1] = v39;
    sub_C8D5F0((__int64)&v103, v82, (unsigned int)v104 + 1LL, 0x10u, v37, v39);
    v40 = (unsigned int)v104;
    v39 = v80.m128i_i64[1];
    v41 = v80.m128i_i64[0];
  }
  v42 = (__int64 *)&v103[16 * v40];
  *v42 = v41;
  v42[1] = v39;
  LODWORD(v42) = *(_DWORD *)(a2 + 4);
  LODWORD(v104) = v104 + 1;
  v43 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - ((unsigned int)v42 & 0x7FFFFFF))));
  v45 = (unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16LL * v44);
  v46 = *(_QWORD *)(v43 + 96);
  if ( *(_DWORD *)(v46 + 32) <= 0x40u )
    v47 = *(_QWORD *)(v46 + 24);
  else
    v47 = **(_QWORD **)(v46 + 24);
  v48 = sub_3400BD0(*(_QWORD *)(a1 + 864), v47, (unsigned int)&v93, *v45, *((_QWORD *)v45 + 1), 1, 0);
  v50 = v49;
  v51 = (unsigned int)v104;
  v52 = v48;
  if ( (unsigned __int64)(unsigned int)v104 + 1 > HIDWORD(v104) )
  {
    v80.m128i_i64[0] = v48;
    v80.m128i_i64[1] = v50;
    sub_C8D5F0((__int64)&v103, v82, (unsigned int)v104 + 1LL, 0x10u, v48, v50);
    v51 = (unsigned int)v104;
    v50 = v80.m128i_i64[1];
    v52 = v80.m128i_i64[0];
  }
  v53 = (__int64 *)&v103[16 * v51];
  v53[1] = v50;
  *v53 = v52;
  LODWORD(v104) = v104 + 1;
  sub_33AEA10((unsigned __int8 *)a2, 2u, (__int64)&v103, a1);
  v54 = sub_33E5110(*(_QWORD *)(a1 + 864), 1, 0, 262, 0);
  *((_QWORD *)&v76 + 1) = (unsigned int)v104;
  *(_QWORD *)&v76 = v103;
  v57 = sub_3411630(*(_QWORD *)(a1 + 864), 393, (unsigned int)&v93, v54, v55, v56, v76);
  v58 = *(_QWORD *)(a1 + 864);
  v90 = v59;
  v60 = (unsigned int)v59;
  v61 = v57;
  v89 = v57;
  v62.m128i_i64[0] = sub_3400D50(v58, 0, &v93, 1);
  v80 = v62;
  v63.m128i_i64[0] = sub_3400D50(v58, 0, &v93, 1);
  v79 = v63;
  v64 = sub_33E5110(v58, 1, 0, 262, 0);
  v67 = _mm_load_si128(&v79);
  v68 = _mm_load_si128(&v80);
  v95 = &v97;
  v69 = 3;
  v97 = v61;
  v98 = v60;
  v96 = 0x400000003LL;
  v99 = v67;
  v100 = v68;
  if ( v61 )
  {
    v101 = v61;
    v69 = 4;
    v102 = 1;
    LODWORD(v96) = 4;
  }
  *((_QWORD *)&v77 + 1) = v69;
  *(_QWORD *)&v77 = &v97;
  v80.m128i_i64[0] = (__int64)&v97;
  v70 = sub_3411630(v58, 316, (unsigned int)&v93, v64, v65, v66, v77);
  v72 = v71;
  if ( v95 != (_QWORD *)v80.m128i_i64[0] )
    _libc_free((unsigned __int64)v95);
  v88 = v72;
  v73 = *(_QWORD *)(a1 + 864);
  v87 = v70;
  if ( v70 )
  {
    nullsub_1875(v70, v73, 0);
    v86 = (unsigned int)v72;
    v85 = v70;
    *(_QWORD *)(v73 + 384) = v70;
    *(_DWORD *)(v73 + 392) = v86;
    sub_33E2B60(v73, 0);
  }
  else
  {
    v84 = (unsigned int)v72;
    v83 = 0;
    *(_QWORD *)(v73 + 384) = 0;
    *(_DWORD *)(v73 + 392) = v84;
  }
  v74 = v93;
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 48LL) + 39LL) = 1;
  if ( v74 )
    sub_B91220((__int64)&v93, v74);
  if ( v103 != v82 )
    _libc_free((unsigned __int64)v103);
}
