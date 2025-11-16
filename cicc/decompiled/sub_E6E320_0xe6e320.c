// Function: sub_E6E320
// Address: 0xe6e320
//
__int64 __fastcall sub_E6E320(_QWORD *a1, _BYTE *a2, __int64 a3, char a4, __int16 a5, char a6, __int64 a7)
{
  _QWORD *v8; // rsi
  _QWORD *v9; // rdx
  char v10; // r13
  __m128i *v11; // rsi
  __m128i *v12; // rdx
  char v13; // dl
  __int64 v14; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  __m128i v25; // rax
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 *v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 *v35; // rcx
  __m128i v36; // xmm2
  __int64 v37; // [rsp+0h] [rbp-130h]
  __int64 v38; // [rsp+10h] [rbp-120h]
  char v39; // [rsp+18h] [rbp-118h]
  __int64 v40; // [rsp+18h] [rbp-118h]
  _QWORD *v43; // [rsp+30h] [rbp-100h]
  _QWORD *v44; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-E8h]
  _QWORD v46[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v47; // [rsp+60h] [rbp-D0h]
  __m128i v48; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v49; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+90h] [rbp-A0h]
  __m128i *v51; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-88h]
  __m128i v53; // [rsp+B0h] [rbp-80h] BYREF
  int v54; // [rsp+C0h] [rbp-70h]
  char v55; // [rsp+C4h] [rbp-6Ch]
  __m128i v56; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v57; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v58; // [rsp+F0h] [rbp-40h]
  __int64 v59; // [rsp+F8h] [rbp-38h]

  if ( BYTE4(a7) )
  {
    if ( a2 )
    {
      v44 = v46;
      sub_E62BB0((__int64 *)&v44, a2, (__int64)&a2[a3]);
      v8 = v44;
      v9 = (_QWORD *)((char *)v44 + v45);
    }
    else
    {
      v9 = v46;
      LOBYTE(v46[0]) = 0;
      v44 = v46;
      v8 = v46;
      v45 = 0;
    }
    v10 = 0;
    v51 = &v53;
    sub_E62BB0((__int64 *)&v51, v8, (__int64)v9);
    v55 = 0;
    v54 = a7;
  }
  else
  {
    if ( a2 )
    {
      v48.m128i_i64[0] = (__int64)&v49;
      sub_E62BB0(v48.m128i_i64, a2, (__int64)&a2[a3]);
      v11 = (__m128i *)v48.m128i_i64[0];
      v12 = (__m128i *)(v48.m128i_i64[0] + v48.m128i_i64[1]);
    }
    else
    {
      v12 = &v49;
      v49.m128i_i8[0] = 0;
      v48 = (__m128i)(unsigned __int64)&v49;
      v11 = &v49;
    }
    v10 = 1;
    v51 = &v53;
    sub_E62BB0((__int64 *)&v51, v11, (__int64)v12);
    v55 = 1;
    LOBYTE(v54) = a5;
  }
  v56.m128i_i64[0] = (__int64)&v57;
  if ( v51 == &v53 )
  {
    v57 = _mm_load_si128(&v53);
  }
  else
  {
    v56.m128i_i64[0] = (__int64)v51;
    v57.m128i_i64[0] = v53.m128i_i64[0];
  }
  v51 = &v53;
  v56.m128i_i64[1] = v52;
  v52 = 0;
  v53.m128i_i8[0] = 0;
  LODWORD(v58) = v54;
  BYTE4(v58) = v10;
  v59 = 0;
  v43 = (_QWORD *)sub_E6AC70(a1 + 271, &v56);
  v39 = v13;
  if ( (__m128i *)v56.m128i_i64[0] != &v57 )
    j_j___libc_free_0(v56.m128i_i64[0], v57.m128i_i64[0] + 1);
  if ( v51 != &v53 )
    j_j___libc_free_0(v51, v53.m128i_i64[0] + 1);
  if ( v10 && (__m128i *)v48.m128i_i64[0] != &v49 )
    j_j___libc_free_0(v48.m128i_i64[0], v49.m128i_i64[0] + 1);
  if ( BYTE4(a7) && v44 != v46 )
    j_j___libc_free_0(v44, v46[0] + 1LL);
  if ( v39 )
  {
    v16 = v43[5];
    v40 = v43[4];
    v38 = v16;
    if ( BYTE4(a7) )
    {
      v56.m128i_i64[0] = v43[4];
      LOWORD(v58) = 261;
      v56.m128i_i64[1] = v16;
      v17 = sub_E6C460((__int64)a1, (const char **)&v56);
      v18 = v17;
      if ( (*(_BYTE *)(v17 + 8) & 1) != 0 )
      {
        v19 = *(__int64 **)(v17 - 8);
        v20 = *v19;
        v21 = (__int64)(v19 + 3);
        v22 = (__int64)v19 + *v19 + 23;
      }
      else
      {
        v22 = -1;
        v20 = 0;
        v21 = 0;
      }
      v56.m128i_i64[0] = v21;
      v56.m128i_i64[1] = v20;
      if ( *(_BYTE *)v22 == 93 )
      {
        LOBYTE(v51) = 91;
        v34 = sub_C93460(v56.m128i_i64, &v51, 1u);
        LODWORD(v21) = v56.m128i_i32[0];
        LODWORD(v20) = v34;
        if ( v34 == -1 )
        {
          LODWORD(v20) = v56.m128i_i32[2];
        }
        else if ( v56.m128i_i64[1] <= v34 )
        {
          LODWORD(v20) = v56.m128i_i32[2];
        }
      }
      v23 = a1[132];
      a1[142] += 192LL;
      v14 = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[133] >= (unsigned __int64)(v14 + 192) && v23 )
        a1[132] = v14 + 192;
      else
        v14 = sub_9D1E70((__int64)(a1 + 132), 192, 192, 3);
      sub_E92760(v14, 5, v21, v20, (unsigned __int8)(a4 - 2) <= 1u, 0, v18);
      *(_BYTE *)(v14 + 150) = 0;
      *(_QWORD *)(v14 + 152) = v18;
      *(_BYTE *)(v14 + 180) = 1;
      *(_QWORD *)v14 = &unk_49E3698;
      *(_QWORD *)(v14 + 160) = v40;
      *(_QWORD *)(v14 + 168) = v38;
      *(_DWORD *)(v14 + 176) = a7;
      *(_BYTE *)(v14 + 184) = a6;
      *(_BYTE *)(v14 + 188) = a4;
      sub_EA1880(v18, v14);
      *(_BYTE *)(v14 + 32) = 5;
      v43[9] = v14;
      v24 = sub_E6B260(a1, v14);
    }
    else
    {
      v51 = (__m128i *)"]";
      LOWORD(v54) = 259;
      v25.m128i_i64[0] = sub_1060220((unsigned __int8)a5);
      v49 = v25;
      v26 = v54;
      v44 = (_QWORD *)v40;
      v47 = 773;
      v45 = v38;
      v46[0] = "[";
      v48.m128i_i64[0] = (__int64)&v44;
      LOWORD(v50) = 1282;
      if ( (_BYTE)v54 )
      {
        if ( (_BYTE)v54 == 1 )
        {
          v36 = _mm_load_si128(&v49);
          v56 = _mm_load_si128(&v48);
          v58 = v50;
          v57 = v36;
        }
        else
        {
          if ( BYTE1(v54) == 1 )
          {
            v35 = (__int64 *)v51;
            v37 = v52;
          }
          else
          {
            v35 = (__int64 *)&v51;
            v26 = 2;
          }
          v57.m128i_i64[0] = (__int64)v35;
          v56.m128i_i64[0] = (__int64)&v48;
          v57.m128i_i64[1] = v37;
          LOBYTE(v58) = 2;
          BYTE1(v58) = v26;
        }
      }
      else
      {
        LOWORD(v58) = 256;
      }
      v27 = sub_E6C460((__int64)a1, (const char **)&v56);
      v18 = v27;
      if ( (*(_BYTE *)(v27 + 8) & 1) != 0 )
      {
        v32 = *(__int64 **)(v27 - 8);
        v29 = *v32;
        v30 = (__int64)(v32 + 3);
        v28 = (__int64)v32 + *v32 + 23;
      }
      else
      {
        v28 = -1;
        v29 = 0;
        v30 = 0;
      }
      v56.m128i_i64[0] = v30;
      v56.m128i_i64[1] = v29;
      if ( *(_BYTE *)v28 == 93 )
      {
        LOBYTE(v51) = 91;
        v33 = sub_C93460(v56.m128i_i64, &v51, 1u);
        LODWORD(v30) = v56.m128i_i32[0];
        LODWORD(v29) = v33;
        if ( v33 == -1 )
        {
          LODWORD(v29) = v56.m128i_i32[2];
        }
        else if ( v56.m128i_i64[1] <= v33 )
        {
          LODWORD(v29) = v56.m128i_i32[2];
        }
      }
      a1[142] += 192LL;
      v31 = a1[132];
      v14 = (v31 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[133] >= (unsigned __int64)(v14 + 192) && v31 )
        a1[132] = v14 + 192;
      else
        v14 = sub_9D1E70((__int64)(a1 + 132), 192, 192, 3);
      sub_E92760(v14, 5, v30, v29, (unsigned __int8)(a4 - 2) <= 1u, HIBYTE(a5) == 3 && (_BYTE)a5 != 16, 0);
      *(_BYTE *)(v14 + 150) = 1;
      *(_BYTE *)(v14 + 148) = a5;
      *(_QWORD *)v14 = &unk_49E3698;
      *(_BYTE *)(v14 + 149) = HIBYTE(a5);
      *(_QWORD *)(v14 + 160) = v40;
      *(_QWORD *)(v14 + 152) = v18;
      *(_QWORD *)(v14 + 168) = v38;
      *(_BYTE *)(v14 + 180) = 0;
      *(_BYTE *)(v14 + 184) = a6;
      *(_BYTE *)(v14 + 188) = a4;
      sub_EA1880(v18, v14);
      *(_WORD *)(v18 + 32) = 363;
      if ( HIBYTE(a5) )
      {
        if ( (_BYTE)a5 )
        {
          *(_BYTE *)(v14 + 32) = 2;
          v43[9] = v14;
          sub_E6B260(a1, v14);
          return v14;
        }
        *(_BYTE *)(v14 + 32) = 5;
        v43[9] = v14;
        v24 = sub_E6B260(a1, v14);
      }
      else
      {
        v43[9] = v14;
        v24 = sub_E6B260(a1, v14);
        if ( (_BYTE)a5 )
          return v14;
      }
    }
    *(_QWORD *)v18 = v24;
  }
  else
  {
    v14 = v43[9];
    if ( a6 != *(_BYTE *)(v14 + 184) )
      sub_C64ED0("section's multiply symbols policy does not match", 1u);
  }
  return v14;
}
