// Function: sub_2078580
// Address: 0x2078580
//
void __fastcall sub_2078580(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rax
  int v23; // edx
  const void ***v24; // r15
  int v25; // eax
  __int64 *v26; // r11
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // r15
  __int64 *v31; // r13
  __int64 *v32; // rax
  unsigned __int8 *v33; // rdi
  bool v34; // al
  __int64 *v35; // r11
  bool v36; // zf
  int v37; // eax
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int128 v46; // [rsp-10h] [rbp-130h]
  __int64 *v47; // [rsp+8h] [rbp-118h]
  __int64 v48; // [rsp+10h] [rbp-110h]
  unsigned int v49; // [rsp+1Ch] [rbp-104h]
  __int64 v50; // [rsp+28h] [rbp-F8h]
  int v51; // [rsp+28h] [rbp-F8h]
  __int64 v52; // [rsp+50h] [rbp-D0h] BYREF
  int v53; // [rsp+58h] [rbp-C8h]
  __int64 v54; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+68h] [rbp-B8h]
  __int64 *v56; // [rsp+70h] [rbp-B0h]
  __int64 v57; // [rsp+78h] [rbp-A8h]
  __int64 *v58; // [rsp+80h] [rbp-A0h]
  __int64 v59; // [rsp+88h] [rbp-98h]
  __int64 *v60; // [rsp+90h] [rbp-90h]
  __int64 v61; // [rsp+98h] [rbp-88h]
  unsigned __int8 *v62; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v63; // [rsp+A8h] [rbp-78h]
  _BYTE v64[112]; // [rsp+B0h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a1 + 536);
  v8 = *(_QWORD *)a1;
  v52 = 0;
  v53 = v7;
  if ( v8 )
  {
    if ( &v52 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v52 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v52, v9, 2);
    }
  }
  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) )
    BUG();
  v49 = *(_DWORD *)&asc_4307A60[4 * (*(_DWORD *)(v10 + 36) - 57)];
  v48 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  v11 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v13 = v12;
  v14 = *(_QWORD *)a2;
  v62 = v64;
  v63 = 0x400000000LL;
  v50 = v14;
  v15 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  sub_20C7CE0(v48, v15, v50, &v62, 0, 0);
  v18 = (unsigned int)v63;
  if ( (unsigned int)v63 >= HIDWORD(v63) )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v16, v17);
    v18 = (unsigned int)v63;
  }
  v19 = &v62[16 * v18];
  *(_QWORD *)v19 = 1;
  v20 = v62;
  *((_QWORD *)v19 + 1) = 0;
  v21 = *(_QWORD *)(a1 + 552);
  LODWORD(v63) = v63 + 1;
  v22 = sub_1D25C30(v21, v20, (unsigned int)v63);
  v51 = v23;
  v24 = (const void ***)v22;
  if ( sub_1602320(a2) )
  {
    v25 = *(_DWORD *)(a2 + 20);
    v26 = *(__int64 **)(a1 + 552);
    v54 = (__int64)v11;
    v55 = v13;
    v47 = v26;
    v56 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (v25 & 0xFFFFFFF)), a3, a4, a5);
    v57 = v27;
    v28 = 2;
  }
  else
  {
    v34 = sub_1602360(a2);
    v35 = *(__int64 **)(a1 + 552);
    v54 = (__int64)v11;
    v36 = !v34;
    v55 = v13;
    v37 = *(_DWORD *)(a2 + 20);
    v47 = v35;
    if ( v36 )
    {
      v43 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (v37 & 0xFFFFFFF)), a3, a4, a5);
      v57 = v44;
      LODWORD(v44) = *(_DWORD *)(a2 + 20);
      v56 = v43;
      v58 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1 - (v44 & 0xFFFFFFF))), a3, a4, a5);
      v59 = v45;
      v28 = 3;
    }
    else
    {
      v38 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (v37 & 0xFFFFFFF)), a3, a4, a5);
      v57 = v39;
      LODWORD(v39) = *(_DWORD *)(a2 + 20);
      v56 = v38;
      v40 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1 - (v39 & 0xFFFFFFF))), a3, a4, a5);
      v59 = v41;
      LODWORD(v41) = *(_DWORD *)(a2 + 20);
      v58 = v40;
      v60 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (2 - (v41 & 0xFFFFFFF))), a3, a4, a5);
      v61 = v42;
      v28 = 4;
    }
  }
  *((_QWORD *)&v46 + 1) = v28;
  *(_QWORD *)&v46 = &v54;
  v29 = sub_1D36D80(
          v47,
          v49,
          (__int64)&v52,
          v24,
          v51,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          (__int64)&v54,
          v46);
  v30 = *(_QWORD *)(a1 + 552);
  v31 = v29;
  if ( v29 )
  {
    nullsub_686();
    *(_QWORD *)(v30 + 176) = v31;
    *(_DWORD *)(v30 + 184) = 1;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v30 + 176) = 0;
    *(_DWORD *)(v30 + 184) = 1;
  }
  v54 = a2;
  v32 = sub_205F5C0(a1 + 8, &v54);
  v33 = v62;
  v32[1] = (__int64)v31;
  *((_DWORD *)v32 + 4) = 0;
  if ( v33 != v64 )
    _libc_free((unsigned __int64)v33);
  if ( v52 )
    sub_161E7C0((__int64)&v52, v52);
}
