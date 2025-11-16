// Function: sub_23A27F0
// Address: 0x23a27f0
//
void __fastcall sub_23A27F0(
        __int64 a1,
        unsigned __int64 *a2,
        char a3,
        unsigned __int8 a4,
        char a5,
        unsigned __int64 *a6,
        __int64 a7,
        __int64 *a8)
{
  int v12; // r14d
  __int64 v13; // rax
  _BYTE *v14; // rdx
  bool v15; // zf
  _BYTE *v16; // rsi
  char v17; // al
  unsigned __int8 v18; // cl
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // rcx
  __m128i *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __m128i *v30; // rax
  __int64 v31; // [rsp+10h] [rbp-130h] BYREF
  __int64 v32; // [rsp+18h] [rbp-128h] BYREF
  __int64 v33[4]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v34; // [rsp+40h] [rbp-100h] BYREF
  _BYTE *v35; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v36; // [rsp+50h] [rbp-F0h]
  _BYTE v37[24]; // [rsp+58h] [rbp-E8h] BYREF
  __m128i *v38; // [rsp+70h] [rbp-D0h] BYREF
  __int8 *v39; // [rsp+78h] [rbp-C8h] BYREF
  __m128i v40; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v41; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+98h] [rbp-A8h]
  __m128i v43; // [rsp+A0h] [rbp-A0h] BYREF
  char v44; // [rsp+B0h] [rbp-90h]
  __int64 v45; // [rsp+B8h] [rbp-88h]
  __m128i *v46; // [rsp+C0h] [rbp-80h] BYREF
  __int64 *v47; // [rsp+C8h] [rbp-78h] BYREF
  __m128i v48; // [rsp+D0h] [rbp-70h] BYREF
  __m128i *v49; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v50; // [rsp+E8h] [rbp-58h]
  __m128i v51; // [rsp+F0h] [rbp-50h] BYREF
  char v52; // [rsp+100h] [rbp-40h]
  __int64 v53; // [rsp+108h] [rbp-38h]

  if ( a3 )
  {
    v12 = 1 - ((a4 == 0) - 1);
    v13 = sub_22077B0(0x10u);
    if ( v13 )
    {
      *(_DWORD *)(v13 + 8) = v12;
      *(_QWORD *)v13 = &unk_4A0D078;
    }
    v46 = (__m128i *)v13;
    sub_23A2230(a2, (unsigned __int64 *)&v46);
    sub_23501E0((__int64 *)&v46);
    v14 = v37;
    v15 = a6[1] == 0;
    LODWORD(v34) = 0;
    BYTE4(v34) = 0;
    v35 = v37;
    v36 = 0;
    v37[0] = 0;
    if ( v15 )
    {
      v16 = v37;
      v17 = 0;
      v18 = 0;
    }
    else
    {
      sub_2240AE0((unsigned __int64 *)&v35, a6);
      v16 = v35;
      v18 = v34;
      v17 = BYTE4(v34);
      v14 = &v35[v36];
    }
    BYTE2(v34) = a5;
    BYTE2(v38) = a5;
    BYTE4(v38) = v17;
    LOWORD(v38) = v18;
    BYTE3(v34) = a4;
    BYTE3(v38) = a4;
    v39 = &v40.m128i_i8[8];
    BYTE1(v34) = 0;
    sub_239EBB0((__int64 *)&v39, v16, (__int64)v14);
    LOBYTE(v42) = a4;
    LODWORD(v46) = (_DWORD)v38;
    BYTE4(v46) = BYTE4(v38);
    v47 = &v48.m128i_i64[1];
    sub_239EBB0((__int64 *)&v47, v39, (__int64)&v39[v40.m128i_i64[0]]);
    LOBYTE(v50) = v42;
    v19 = sub_22077B0(0x38u);
    v20 = v19;
    if ( v19 )
    {
      v21 = v47;
      v22 = v48.m128i_i64[0];
      *(_QWORD *)v19 = &unk_4A0D578;
      *(_DWORD *)(v19 + 8) = (_DWORD)v46;
      *(_BYTE *)(v19 + 12) = BYTE4(v46);
      *(_QWORD *)(v19 + 16) = v19 + 32;
      sub_239EBB0((__int64 *)(v19 + 16), v21, (__int64)v21 + v22);
      *(_BYTE *)(v20 + 48) = v50;
    }
    v33[0] = v20;
    sub_23A2230(a2, (unsigned __int64 *)v33);
    sub_23501E0(v33);
    sub_2240A30((unsigned __int64 *)&v47);
    sub_2240A30((unsigned __int64 *)&v39);
    sub_2240A30((unsigned __int64 *)&v35);
  }
  else
  {
    v23 = *a8;
    v31 = v23;
    if ( v23 )
      _InterlockedAdd((volatile signed __int32 *)(v23 + 8), 1u);
    sub_2241BD0(&v34, a7);
    sub_2241BD0(v33, (__int64)a6);
    sub_24A9AC0(&v38, v33, &v34, a4, &v31);
    v46 = &v48;
    if ( v38 == &v40 )
    {
      v48 = _mm_load_si128(&v40);
    }
    else
    {
      v46 = v38;
      v48.m128i_i64[0] = v40.m128i_i64[0];
    }
    v24 = (__int64 *)v39;
    v38 = &v40;
    v39 = 0;
    v47 = v24;
    v40.m128i_i8[0] = 0;
    v49 = &v51;
    if ( v41 == &v43 )
    {
      v51 = _mm_load_si128(&v43);
    }
    else
    {
      v49 = v41;
      v51.m128i_i64[0] = v43.m128i_i64[0];
    }
    v41 = &v43;
    v52 = v44;
    v50 = v42;
    v42 = 0;
    v43.m128i_i8[0] = 0;
    v53 = v45;
    v45 = 0;
    v25 = sub_22077B0(0x58u);
    if ( v25 )
    {
      *(_QWORD *)v25 = &unk_4A0DB78;
      *(_QWORD *)(v25 + 8) = v25 + 24;
      if ( v46 == &v48 )
      {
        *(__m128i *)(v25 + 24) = _mm_load_si128(&v48);
      }
      else
      {
        *(_QWORD *)(v25 + 8) = v46;
        *(_QWORD *)(v25 + 24) = v48.m128i_i64[0];
      }
      v46 = &v48;
      v26 = v47;
      *(_QWORD *)(v25 + 40) = v25 + 56;
      v27 = v49;
      *(_QWORD *)(v25 + 16) = v26;
      v47 = 0;
      v48.m128i_i8[0] = 0;
      if ( v27 == &v51 )
      {
        *(__m128i *)(v25 + 56) = _mm_load_si128(&v51);
      }
      else
      {
        *(_QWORD *)(v25 + 40) = v27;
        *(_QWORD *)(v25 + 56) = v51.m128i_i64[0];
      }
      v28 = v50;
      v49 = &v51;
      v50 = 0;
      *(_QWORD *)(v25 + 48) = v28;
      v51.m128i_i8[0] = 0;
      *(_BYTE *)(v25 + 72) = v52;
      v29 = v53;
      v53 = 0;
      *(_QWORD *)(v25 + 80) = v29;
    }
    v32 = v25;
    sub_23A2230(a2, (unsigned __int64 *)&v32);
    sub_23501E0(&v32);
    if ( v53 )
      sub_23569D0((volatile signed __int32 *)(v53 + 8));
    sub_2240A30((unsigned __int64 *)&v49);
    sub_2240A30((unsigned __int64 *)&v46);
    if ( v45 )
      sub_23569D0((volatile signed __int32 *)(v45 + 8));
    sub_2240A30((unsigned __int64 *)&v41);
    sub_2240A30((unsigned __int64 *)&v38);
    sub_2240A30((unsigned __int64 *)v33);
    sub_2240A30((unsigned __int64 *)&v34);
    if ( v31 )
      sub_23569D0((volatile signed __int32 *)(v31 + 8));
    v30 = (__m128i *)sub_22077B0(0x10u);
    if ( v30 )
      v30->m128i_i64[0] = (__int64)&unk_4A0CB38;
    v46 = v30;
    sub_23A2230(a2, (unsigned __int64 *)&v46);
    if ( v46 )
      (*(void (__fastcall **)(__m128i *))(v46->m128i_i64[0] + 8))(v46);
  }
}
