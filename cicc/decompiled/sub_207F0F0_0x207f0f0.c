// Function: sub_207F0F0
// Address: 0x207f0f0
//
void __fastcall sub_207F0F0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // r14
  unsigned int v12; // edx
  __int64 *v13; // rax
  int v14; // edx
  __int64 v15; // rax
  int v16; // edx
  const void ***v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 *v22; // r15
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // edx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 *v32; // rax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // edx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 *v40; // rax
  int v41; // r8d
  int v42; // r9d
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  int v49; // edx
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 v53; // r13
  const void ***v54; // rax
  int v55; // edx
  __int64 v56; // r9
  __int64 v57; // rsi
  __int64 *v58; // r15
  int v59; // edx
  int v60; // r13d
  __int64 v61; // r14
  __int64 v62; // rsi
  __int128 v63; // [rsp-10h] [rbp-310h]
  __int128 v64; // [rsp-10h] [rbp-310h]
  unsigned __int64 v65; // [rsp+0h] [rbp-300h]
  __int64 v66; // [rsp+0h] [rbp-300h]
  __int64 v67; // [rsp+0h] [rbp-300h]
  int v68; // [rsp+8h] [rbp-2F8h]
  __int64 v69; // [rsp+8h] [rbp-2F8h]
  __int64 v70; // [rsp+8h] [rbp-2F8h]
  __int64 v71; // [rsp+10h] [rbp-2F0h]
  int v72; // [rsp+18h] [rbp-2E8h]
  __int64 v73; // [rsp+18h] [rbp-2E8h]
  __int64 v74; // [rsp+20h] [rbp-2E0h]
  __int64 v75; // [rsp+60h] [rbp-2A0h] BYREF
  int v76; // [rsp+68h] [rbp-298h]
  __int64 *v77; // [rsp+70h] [rbp-290h] BYREF
  __int64 v78; // [rsp+78h] [rbp-288h]
  __int64 v79; // [rsp+80h] [rbp-280h] BYREF
  __int64 v80; // [rsp+88h] [rbp-278h]
  __int64 v81; // [rsp+90h] [rbp-270h]
  __int64 v82; // [rsp+98h] [rbp-268h]
  __int64 v83; // [rsp+A0h] [rbp-260h]
  __int64 v84; // [rsp+A8h] [rbp-258h]
  __int64 v85; // [rsp+B0h] [rbp-250h]
  __int64 v86; // [rsp+B8h] [rbp-248h]
  __int64 *v87; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-238h]
  _BYTE v89[560]; // [rsp+D0h] [rbp-230h] BYREF

  v7 = *(_DWORD *)(a1 + 536);
  v87 = (__int64 *)v89;
  v88 = 0x2000000000LL;
  v8 = *(_QWORD *)a1;
  v75 = 0;
  v76 = v7;
  if ( v8 )
  {
    if ( &v75 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v75 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v75, v9, 2);
    }
  }
  sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
  v10 = sub_1D38E70(*(_QWORD *)(a1 + 552), 0, (__int64)&v75, 1u, a3, *(double *)a4.m128i_i64, a5);
  v11 = *(__int64 **)(a1 + 552);
  v71 = v10;
  v74 = v12;
  v13 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v68 = v14;
  v65 = (unsigned __int64)v13;
  v15 = sub_1D252B0((__int64)v11, 1, 0, 111, 0);
  v72 = v16;
  v17 = (const void ***)v15;
  v77 = (__int64 *)v65;
  LODWORD(v78) = v68;
  v18 = sub_1D38E70((__int64)v11, 0, (__int64)&v75, 1u, a3, *(double *)a4.m128i_i64, a5);
  v80 = v19;
  v79 = v18;
  v81 = sub_1D38E70((__int64)v11, 0, (__int64)&v75, 1u, a3, *(double *)a4.m128i_i64, a5);
  v82 = v20;
  *((_QWORD *)&v63 + 1) = 3;
  *(_QWORD *)&v63 = &v77;
  v22 = sub_1D36D80(v11, 201, (__int64)&v75, v17, v72, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v21, v63);
  v73 = v23;
  v24 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a3, a4, a5);
  v25 = *(_QWORD *)(a1 + 552);
  v26 = v24[11];
  if ( *(_DWORD *)(v26 + 32) <= 0x40u )
    v27 = *(_QWORD *)(v26 + 24);
  else
    v27 = **(_QWORD **)(v26 + 24);
  v29 = sub_1D38BB0(v25, v27, (__int64)&v75, 6, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v30 = v28;
  v31 = (unsigned int)v88;
  if ( (unsigned int)v88 >= HIDWORD(v88) )
  {
    v67 = v29;
    v70 = v28;
    sub_16CD150((__int64)&v87, v89, 0, 16, v29, v28);
    v31 = (unsigned int)v88;
    v29 = v67;
    v30 = v70;
  }
  v32 = &v87[2 * v31];
  *v32 = v29;
  v32[1] = v30;
  v33 = *(_DWORD *)(a2 + 20);
  LODWORD(v88) = v88 + 1;
  v34 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1LL - (v33 & 0xFFFFFFF))), a3, a4, a5)[11];
  if ( *(_DWORD *)(v34 + 32) <= 0x40u )
    v35 = *(_QWORD *)(v34 + 24);
  else
    v35 = **(_QWORD **)(v34 + 24);
  v37 = sub_1D38BB0(*(_QWORD *)(a1 + 552), v35, (__int64)&v75, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v38 = v36;
  v39 = (unsigned int)v88;
  if ( (unsigned int)v88 >= HIDWORD(v88) )
  {
    v66 = v37;
    v69 = v36;
    sub_16CD150((__int64)&v87, v89, 0, 16, v37, v36);
    v39 = (unsigned int)v88;
    v37 = v66;
    v38 = v69;
  }
  v40 = &v87[2 * v39];
  *v40 = v37;
  v40[1] = v38;
  LODWORD(v88) = v88 + 1;
  sub_207EC30(a2 | 4, 2u, (__int64)&v75, (__int64)&v87, a1, a3, a4, a5);
  v43 = (unsigned int)v88;
  if ( (unsigned int)v88 >= HIDWORD(v88) )
  {
    sub_16CD150((__int64)&v87, v89, 0, 16, v41, v42);
    v43 = (unsigned int)v88;
  }
  v44 = &v87[2 * v43];
  *v44 = (__int64)v22;
  v44[1] = v73;
  v45 = (unsigned int)(v88 + 1);
  LODWORD(v88) = v45;
  if ( HIDWORD(v88) <= (unsigned int)v45 )
  {
    sub_16CD150((__int64)&v87, v89, 0, 16, v41, v42);
    v45 = (unsigned int)v88;
  }
  v46 = &v87[2 * v45];
  *v46 = (__int64)v22;
  v46[1] = 1;
  v47 = *(_QWORD *)(a1 + 552);
  LODWORD(v88) = v88 + 1;
  v48 = sub_1D252B0(v47, 1, 0, 111, 0);
  v51 = sub_1D23DE0(*(_QWORD **)(a1 + 552), 19, (__int64)&v75, v48, v49, v50, v87, (unsigned int)v88);
  v52 = *(__int64 **)(a1 + 552);
  v53 = v51;
  v54 = (const void ***)sub_1D252B0((__int64)v52, 1, 0, 111, 0);
  v77 = &v79;
  v79 = v53;
  v82 = v74;
  v81 = v71;
  v83 = v71;
  v84 = v74;
  v57 = 3;
  v80 = 0;
  v78 = 0x400000003LL;
  if ( v53 )
  {
    v85 = v53;
    v57 = 4;
    v86 = 1;
    LODWORD(v78) = 4;
  }
  *((_QWORD *)&v64 + 1) = v57;
  *(_QWORD *)&v64 = &v79;
  v58 = sub_1D36D80(v52, 203, (__int64)&v75, v54, v55, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v56, v64);
  v60 = v59;
  if ( v77 != &v79 )
    _libc_free((unsigned __int64)v77);
  v61 = *(_QWORD *)(a1 + 552);
  if ( v58 )
  {
    nullsub_686();
    *(_QWORD *)(v61 + 176) = v58;
    *(_DWORD *)(v61 + 184) = v60;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v61 + 176) = 0;
    *(_DWORD *)(v61 + 184) = v60;
  }
  v62 = v75;
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 56LL) + 39LL) = 1;
  if ( v62 )
    sub_161E7C0((__int64)&v75, v62);
  if ( v87 != (__int64 *)v89 )
    _libc_free((unsigned __int64)v87);
}
