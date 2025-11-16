// Function: sub_23A2D30
// Address: 0x23a2d30
//
void __fastcall sub_23A2D30(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        char a4,
        unsigned __int8 a5,
        char a6,
        unsigned __int64 *a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v13; // rax
  _BYTE *v14; // rdx
  bool v15; // zf
  _BYTE *v16; // rsi
  char v17; // cl
  char v18; // al
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
  int v31; // [rsp+0h] [rbp-140h]
  char v32; // [rsp+8h] [rbp-138h]
  __int64 v33; // [rsp+10h] [rbp-130h] BYREF
  __int64 v34; // [rsp+18h] [rbp-128h] BYREF
  __int64 v35[4]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v36; // [rsp+40h] [rbp-100h] BYREF
  _BYTE *v37; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v38; // [rsp+50h] [rbp-F0h]
  _BYTE v39[24]; // [rsp+58h] [rbp-E8h] BYREF
  __m128i *v40; // [rsp+70h] [rbp-D0h] BYREF
  __int8 *v41; // [rsp+78h] [rbp-C8h] BYREF
  __m128i v42; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v43; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v44; // [rsp+98h] [rbp-A8h]
  __m128i v45; // [rsp+A0h] [rbp-A0h] BYREF
  char v46; // [rsp+B0h] [rbp-90h]
  __int64 v47; // [rsp+B8h] [rbp-88h]
  __m128i *v48; // [rsp+C0h] [rbp-80h] BYREF
  __int64 *v49; // [rsp+C8h] [rbp-78h] BYREF
  __m128i v50; // [rsp+D0h] [rbp-70h] BYREF
  __m128i *v51; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v52; // [rsp+E8h] [rbp-58h]
  __m128i v53; // [rsp+F0h] [rbp-50h] BYREF
  char v54; // [rsp+100h] [rbp-40h]
  __int64 v55; // [rsp+108h] [rbp-38h]

  v32 = a4;
  if ( a4 )
  {
    v31 = 1 - ((a5 == 0) - 1);
    v13 = sub_22077B0(0x10u);
    if ( v13 )
    {
      *(_DWORD *)(v13 + 8) = v31;
      *(_QWORD *)v13 = &unk_4A0D078;
    }
    v48 = (__m128i *)v13;
    sub_23A2230(a2, (unsigned __int64 *)&v48);
    sub_23501E0((__int64 *)&v48);
    sub_23A2470(a1, a2, a3);
    v14 = v39;
    LODWORD(v36) = 0;
    BYTE4(v36) = 0;
    v15 = a7[1] == 0;
    v37 = v39;
    v38 = 0;
    v39[0] = 0;
    if ( v15 )
    {
      v16 = v39;
      v17 = 0;
    }
    else
    {
      sub_2240AE0((unsigned __int64 *)&v37, a7);
      v16 = v37;
      v17 = v36;
      v14 = &v37[v38];
    }
    v18 = byte_4FDC628;
    BYTE1(v36) = 1;
    BYTE3(v36) = a5;
    if ( byte_4FDC628 )
    {
      BYTE4(v36) = 1;
      BYTE1(v36) = 0;
      v32 = 0;
    }
    else
    {
      v18 = BYTE4(v36);
    }
    BYTE2(v36) = a6;
    LOBYTE(v40) = v17;
    BYTE2(v40) = a6;
    BYTE4(v40) = v18;
    BYTE1(v40) = v32;
    BYTE3(v40) = a5;
    v41 = &v42.m128i_i8[8];
    sub_239EBB0((__int64 *)&v41, v16, (__int64)v14);
    LOBYTE(v44) = a5;
    LODWORD(v48) = (_DWORD)v40;
    BYTE4(v48) = BYTE4(v40);
    v49 = &v50.m128i_i64[1];
    sub_239EBB0((__int64 *)&v49, v41, (__int64)&v41[v42.m128i_i64[0]]);
    LOBYTE(v52) = v44;
    v19 = sub_22077B0(0x38u);
    v20 = v19;
    if ( v19 )
    {
      v21 = v49;
      v22 = v50.m128i_i64[0];
      *(_QWORD *)v19 = &unk_4A0D578;
      *(_DWORD *)(v19 + 8) = (_DWORD)v48;
      *(_BYTE *)(v19 + 12) = BYTE4(v48);
      *(_QWORD *)(v19 + 16) = v19 + 32;
      sub_239EBB0((__int64 *)(v19 + 16), v21, (__int64)v21 + v22);
      *(_BYTE *)(v20 + 48) = v52;
    }
    v35[0] = v20;
    sub_23A2230(a2, (unsigned __int64 *)v35);
    sub_23501E0(v35);
    sub_2240A30((unsigned __int64 *)&v49);
    sub_2240A30((unsigned __int64 *)&v41);
    sub_2240A30((unsigned __int64 *)&v37);
  }
  else
  {
    v23 = *a9;
    v33 = v23;
    if ( v23 )
      _InterlockedAdd((volatile signed __int32 *)(v23 + 8), 1u);
    sub_2241BD0(&v36, a8);
    sub_2241BD0(v35, (__int64)a7);
    sub_24A9AC0(&v40, v35, &v36, a5, &v33);
    v48 = &v50;
    if ( v40 == &v42 )
    {
      v50 = _mm_load_si128(&v42);
    }
    else
    {
      v48 = v40;
      v50.m128i_i64[0] = v42.m128i_i64[0];
    }
    v24 = (__int64 *)v41;
    v40 = &v42;
    v41 = 0;
    v49 = v24;
    v42.m128i_i8[0] = 0;
    v51 = &v53;
    if ( v43 == &v45 )
    {
      v53 = _mm_load_si128(&v45);
    }
    else
    {
      v51 = v43;
      v53.m128i_i64[0] = v45.m128i_i64[0];
    }
    v43 = &v45;
    v54 = v46;
    v52 = v44;
    v44 = 0;
    v45.m128i_i8[0] = 0;
    v55 = v47;
    v47 = 0;
    v25 = sub_22077B0(0x58u);
    if ( v25 )
    {
      *(_QWORD *)v25 = &unk_4A0DB78;
      *(_QWORD *)(v25 + 8) = v25 + 24;
      if ( v48 == &v50 )
      {
        *(__m128i *)(v25 + 24) = _mm_load_si128(&v50);
      }
      else
      {
        *(_QWORD *)(v25 + 8) = v48;
        *(_QWORD *)(v25 + 24) = v50.m128i_i64[0];
      }
      v48 = &v50;
      v26 = v49;
      *(_QWORD *)(v25 + 40) = v25 + 56;
      v27 = v51;
      *(_QWORD *)(v25 + 16) = v26;
      v49 = 0;
      v50.m128i_i8[0] = 0;
      if ( v27 == &v53 )
      {
        *(__m128i *)(v25 + 56) = _mm_load_si128(&v53);
      }
      else
      {
        *(_QWORD *)(v25 + 40) = v27;
        *(_QWORD *)(v25 + 56) = v53.m128i_i64[0];
      }
      v28 = v52;
      v51 = &v53;
      v52 = 0;
      *(_QWORD *)(v25 + 48) = v28;
      v53.m128i_i8[0] = 0;
      *(_BYTE *)(v25 + 72) = v54;
      v29 = v55;
      v55 = 0;
      *(_QWORD *)(v25 + 80) = v29;
    }
    v34 = v25;
    sub_23A2230(a2, (unsigned __int64 *)&v34);
    sub_23501E0(&v34);
    if ( v55 )
      sub_23569D0((volatile signed __int32 *)(v55 + 8));
    sub_2240A30((unsigned __int64 *)&v51);
    sub_2240A30((unsigned __int64 *)&v48);
    if ( v47 )
      sub_23569D0((volatile signed __int32 *)(v47 + 8));
    sub_2240A30((unsigned __int64 *)&v43);
    sub_2240A30((unsigned __int64 *)&v40);
    sub_2240A30((unsigned __int64 *)v35);
    sub_2240A30((unsigned __int64 *)&v36);
    if ( v33 )
      sub_23569D0((volatile signed __int32 *)(v33 + 8));
    v30 = (__m128i *)sub_22077B0(0x10u);
    if ( v30 )
      v30->m128i_i64[0] = (__int64)&unk_4A0CB38;
    v48 = v30;
    sub_23A2230(a2, (unsigned __int64 *)&v48);
    if ( v48 )
      (*(void (__fastcall **)(__m128i *))(v48->m128i_i64[0] + 8))(v48);
  }
}
