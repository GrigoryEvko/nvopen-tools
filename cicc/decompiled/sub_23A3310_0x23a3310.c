// Function: sub_23A3310
// Address: 0x23a3310
//
__int64 __fastcall sub_23A3310(__int64 a1)
{
  _QWORD *v1; // rax
  char *v2; // rax
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx
  __m128i v8; // xmm0
  __m128i v9; // xmm2
  char *v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rax
  unsigned __int64 *v13; // rsi
  _QWORD *v14; // r14
  _QWORD *v15; // r12
  _QWORD *v16; // r14
  _QWORD *v17; // r12
  __int64 v19; // [rsp+8h] [rbp-3C8h] BYREF
  _QWORD *v20; // [rsp+10h] [rbp-3C0h] BYREF
  _QWORD *v21; // [rsp+18h] [rbp-3B8h]
  __int64 v22; // [rsp+20h] [rbp-3B0h]
  __int64 v23; // [rsp+28h] [rbp-3A8h]
  __int64 v24; // [rsp+30h] [rbp-3A0h]
  _QWORD *v25; // [rsp+40h] [rbp-390h] BYREF
  _QWORD *v26; // [rsp+48h] [rbp-388h]
  __int64 v27; // [rsp+50h] [rbp-380h]
  __int64 v28; // [rsp+58h] [rbp-378h]
  __int64 v29; // [rsp+60h] [rbp-370h]
  __m128i v30; // [rsp+70h] [rbp-360h] BYREF
  char *v31; // [rsp+80h] [rbp-350h]
  __int64 v32; // [rsp+88h] [rbp-348h]
  char v33; // [rsp+90h] [rbp-340h]
  __m128i v34; // [rsp+A0h] [rbp-330h] BYREF
  char *v35; // [rsp+B0h] [rbp-320h]
  __int64 v36; // [rsp+B8h] [rbp-318h]
  int v37; // [rsp+C0h] [rbp-310h]
  char v38; // [rsp+C4h] [rbp-30Ch]
  char v39; // [rsp+C8h] [rbp-308h] BYREF
  __int64 v40; // [rsp+1C8h] [rbp-208h]
  __int64 v41; // [rsp+1D0h] [rbp-200h]
  __int64 v42; // [rsp+1D8h] [rbp-1F8h]
  int v43; // [rsp+1E0h] [rbp-1F0h]
  _QWORD *v44; // [rsp+1E8h] [rbp-1E8h]
  __int64 v45; // [rsp+1F0h] [rbp-1E0h]
  __int64 v46; // [rsp+1F8h] [rbp-1D8h]
  __int64 v47; // [rsp+200h] [rbp-1D0h]
  int v48; // [rsp+208h] [rbp-1C8h]
  __int64 v49; // [rsp+210h] [rbp-1C0h]
  _QWORD v50[7]; // [rsp+218h] [rbp-1B8h] BYREF
  _QWORD v51[4]; // [rsp+250h] [rbp-180h] BYREF
  int v52; // [rsp+270h] [rbp-160h]
  __int64 v53; // [rsp+278h] [rbp-158h]
  char *v54; // [rsp+280h] [rbp-150h]
  __int64 v55; // [rsp+288h] [rbp-148h]
  int v56; // [rsp+290h] [rbp-140h]
  char v57; // [rsp+294h] [rbp-13Ch]
  char v58; // [rsp+298h] [rbp-138h] BYREF

  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v1 = (_QWORD *)sub_22077B0(0x10u);
  if ( v1 )
    *v1 = &unk_4A0CFF8;
  v34.m128i_i64[0] = (__int64)v1;
  sub_23A2230((unsigned __int64 *)&v20, (unsigned __int64 *)&v34);
  sub_23501E0(v34.m128i_i64);
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  sub_24E6490(&v30, 0);
  v2 = v31;
  v3 = _mm_loadu_si128(&v30);
  v4 = _mm_loadu_si128(&v34);
  v5 = v36;
  v31 = 0;
  v35 = v2;
  v30 = v4;
  v36 = v32;
  v32 = v5;
  LOBYTE(v37) = v33;
  v34 = v3;
  v6 = sub_22077B0(0x30u);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 32);
    v8 = _mm_loadu_si128(&v34);
    v9 = _mm_loadu_si128((const __m128i *)(v6 + 8));
    *(_QWORD *)v6 = &unk_4A0EC38;
    v10 = v35;
    v35 = 0;
    *(_QWORD *)(v6 + 24) = v10;
    v11 = v36;
    v36 = v7;
    *(_QWORD *)(v6 + 32) = v11;
    v34 = v9;
    *(_BYTE *)(v6 + 40) = v37;
    *(__m128i *)(v6 + 8) = v8;
  }
  v19 = v6;
  sub_23A32D0((unsigned __int64 *)&v25, (unsigned __int64 *)&v19);
  if ( v19 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
  if ( v35 )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v35)(&v34, &v34, 3);
  if ( v31 )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v31)(&v30, &v30, 3);
  sub_234A9E0(&v34, (unsigned __int64 *)&v25);
  sub_2357280((unsigned __int64 *)&v20, v34.m128i_i64);
  if ( v34.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v34.m128i_i64[0] + 8LL))(v34.m128i_i64[0]);
  v12 = (_QWORD *)sub_22077B0(0x10u);
  if ( v12 )
    *v12 = &unk_4A0CFB8;
  v34.m128i_i64[0] = (__int64)v12;
  sub_23A2230((unsigned __int64 *)&v20, (unsigned __int64 *)&v34);
  sub_23501E0(v34.m128i_i64);
  v35 = &v39;
  v44 = v50;
  v50[1] = v51;
  v54 = &v58;
  v34.m128i_i8[0] = 0;
  v34.m128i_i64[1] = 0;
  v36 = 32;
  v37 = 0;
  v38 = 1;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v45 = 1;
  v46 = 0;
  v47 = 0;
  v48 = 1065353216;
  v49 = 0;
  v50[0] = 0;
  v50[2] = 1;
  v50[3] = 0;
  v50[4] = 0;
  v50[5] = 1065353216;
  v50[6] = 0;
  memset(v51, 0, sizeof(v51));
  v52 = 0;
  v53 = 0;
  v55 = 32;
  v56 = 0;
  v57 = 1;
  sub_23A2670((unsigned __int64 *)&v20, (__int64)&v34);
  sub_233AAF0((__int64)&v34);
  v13 = (unsigned __int64 *)&v20;
  sub_24DC960(a1, &v20);
  v14 = v26;
  v15 = v25;
  if ( v26 != v25 )
  {
    do
    {
      if ( *v15 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v15 + 8LL))(*v15);
      ++v15;
    }
    while ( v14 != v15 );
    v15 = v25;
  }
  if ( v15 )
  {
    v13 = (unsigned __int64 *)(v27 - (_QWORD)v15);
    j_j___libc_free_0((unsigned __int64)v15);
  }
  v16 = v21;
  v17 = v20;
  if ( v21 != v20 )
  {
    do
    {
      if ( *v17 )
        (*(void (__fastcall **)(_QWORD, unsigned __int64 *))(*(_QWORD *)*v17 + 8LL))(*v17, v13);
      ++v17;
    }
    while ( v16 != v17 );
    v17 = v20;
  }
  if ( v17 )
    j_j___libc_free_0((unsigned __int64)v17);
  return a1;
}
