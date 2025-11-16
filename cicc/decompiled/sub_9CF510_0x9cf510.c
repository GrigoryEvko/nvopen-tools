// Function: sub_9CF510
// Address: 0x9cf510
//
__m128i *__fastcall sub_9CF510(__m128i *a1, __int64 a2)
{
  __int64 v4; // rcx
  unsigned __int64 v5; // rax
  __m128i *v6; // rsi
  unsigned __int64 v7; // rax
  __m128i *v8; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int32 v27; // [rsp+10h] [rbp-440h]
  __int32 v28; // [rsp+14h] [rbp-43Ch]
  __int64 v29; // [rsp+28h] [rbp-428h] BYREF
  __int64 v30; // [rsp+30h] [rbp-420h] BYREF
  char v31; // [rsp+38h] [rbp-418h]
  __m128i *v32; // [rsp+40h] [rbp-410h] BYREF
  __int64 v33; // [rsp+48h] [rbp-408h]
  __m128i v34; // [rsp+50h] [rbp-400h] BYREF
  __m128i v35[2]; // [rsp+60h] [rbp-3F0h] BYREF
  char v36; // [rsp+80h] [rbp-3D0h]
  char v37; // [rsp+81h] [rbp-3CFh]
  __m128i v38[2]; // [rsp+90h] [rbp-3C0h] BYREF
  __int16 v39; // [rsp+B0h] [rbp-3A0h]
  __m128i v40[3]; // [rsp+C0h] [rbp-390h] BYREF
  __m128i v41[2]; // [rsp+F0h] [rbp-360h] BYREF
  char v42; // [rsp+110h] [rbp-340h]
  char v43; // [rsp+111h] [rbp-33Fh]
  __m128i v44[3]; // [rsp+120h] [rbp-330h] BYREF
  __m128i v45[2]; // [rsp+150h] [rbp-300h] BYREF
  __int16 v46; // [rsp+170h] [rbp-2E0h]
  __m128i v47[3]; // [rsp+180h] [rbp-2D0h] BYREF
  __m128i v48[2]; // [rsp+1B0h] [rbp-2A0h] BYREF
  char v49; // [rsp+1D0h] [rbp-280h]
  char v50; // [rsp+1D1h] [rbp-27Fh]
  __m128i v51[2]; // [rsp+1E0h] [rbp-270h] BYREF
  char v52; // [rsp+200h] [rbp-250h]
  char v53; // [rsp+201h] [rbp-24Fh]
  __int64 *v54; // [rsp+210h] [rbp-240h] BYREF
  __int64 v55; // [rsp+218h] [rbp-238h]
  _BYTE v56[560]; // [rsp+220h] [rbp-230h] BYREF

  sub_A4DCE0(&v54, a2, 13, 0);
  v5 = (unsigned __int64)v54 & 0xFFFFFFFFFFFFFFFELL;
  if ( ((unsigned __int64)v54 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    a1[2].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v5;
  }
  else
  {
    v33 = 0;
    v54 = (__int64 *)v56;
    v55 = 0x4000000000LL;
    v32 = &v34;
    v34.m128i_i8[0] = 0;
    while ( 1 )
    {
      v6 = (__m128i *)a2;
      sub_9CEA50((__int64)v51, a2, 0, v4);
      if ( (v51[0].m128i_i8[8] & 1) != 0 )
      {
        v51[0].m128i_i8[8] &= ~2u;
        v23 = v51[0].m128i_i64[0];
        v51[0].m128i_i64[0] = 0;
        v48[0].m128i_i64[0] = v23 | 1;
      }
      else
      {
        v48[0].m128i_i64[0] = 1;
        v27 = v51[0].m128i_u32[1];
        v28 = v51[0].m128i_i32[0];
      }
      v7 = v48[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v48[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        a1[2].m128i_i8[0] |= 3u;
        v8 = v32;
        a1->m128i_i64[0] = v7;
        goto LABEL_7;
      }
      if ( v28 == 1 )
      {
        a1[2].m128i_i8[0] = a1[2].m128i_i8[0] & 0xFC | 2;
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( v32 == &v34 )
        {
          a1[1] = _mm_load_si128(&v34);
        }
        else
        {
          a1->m128i_i64[0] = (__int64)v32;
          a1[1].m128i_i64[0] = v34.m128i_i64[0];
        }
        a1->m128i_i64[1] = v33;
        goto LABEL_9;
      }
      if ( v28 != 3 )
      {
        v6 = v51;
        v53 = 1;
        v51[0].m128i_i64[0] = (__int64)"Malformed block";
        v52 = 3;
        sub_9C8190(v48[0].m128i_i64, (__int64)v51);
        v24 = v48[0].m128i_i64[0];
        a1[2].m128i_i8[0] |= 3u;
        v8 = v32;
        a1->m128i_i64[0] = v24 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_7;
      }
      LODWORD(v55) = 0;
      sub_A4B600(&v30, a2, v27, &v54, 0);
      v12 = v31 & 1;
      v4 = (unsigned int)(2 * v12);
      v31 = (2 * v12) | v31 & 0xFD;
      if ( (_BYTE)v12 )
        break;
      if ( (_DWORD)v30 == 1 )
      {
        sub_9C57B0(v54, (unsigned int)v55, (__int64 *)&v32);
        if ( (v31 & 2) != 0 )
          goto LABEL_34;
        if ( (v31 & 1) != 0 && v30 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
      }
      else
      {
        if ( (_DWORD)v30 != 2 )
        {
          v6 = v51;
          v53 = 1;
          v51[0].m128i_i64[0] = (__int64)"Invalid value";
          v52 = 3;
          sub_9C8190(v48[0].m128i_i64, (__int64)v51);
          v25 = v48[0].m128i_i64[0];
          a1[2].m128i_i8[0] |= 3u;
          a1->m128i_i64[0] = v25 & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_20;
        }
        if ( (unsigned int)*v54 )
        {
          v38[0].m128i_i32[0] = *v54;
          v48[0].m128i_i64[0] = (__int64)"'";
          v35[0].m128i_i64[0] = (__int64)"Incompatible epoch: Bitcode '";
          v41[0].m128i_i64[0] = (__int64)"' vs current: '";
          v50 = 1;
          v49 = 3;
          v45[0].m128i_i64[0] = 0;
          v46 = 266;
          v43 = 1;
          v42 = 3;
          v39 = 265;
          v37 = 1;
          v36 = 3;
          sub_9C6370(v40, v35, v38, (__int64)"' vs current: '", v10, v11);
          sub_9C6370(v44, v40, v41, v13, v14, v15);
          sub_9C6370(v47, v44, v45, v16, v17, v18);
          sub_9C6370(v51, v47, v48, v19, v20, v21);
          v6 = v51;
          sub_9C8190(&v29, (__int64)v51);
          v22 = v29;
          a1[2].m128i_i8[0] |= 3u;
          a1->m128i_i64[0] = v22 & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_20;
        }
      }
    }
    v6 = (__m128i *)&v30;
    sub_9C8CD0(v51[0].m128i_i64, &v30);
    v26 = v51[0].m128i_i64[0];
    a1[2].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v26 & 0xFFFFFFFFFFFFFFFELL;
LABEL_20:
    if ( (v31 & 2) != 0 )
LABEL_34:
      sub_9CE230(&v30);
    if ( (v31 & 1) != 0 && v30 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
    v8 = v32;
LABEL_7:
    if ( v8 != &v34 )
    {
      v6 = (__m128i *)(v34.m128i_i64[0] + 1);
      j_j___libc_free_0(v8, v34.m128i_i64[0] + 1);
    }
LABEL_9:
    if ( v54 != (__int64 *)v56 )
      _libc_free(v54, v6);
  }
  return a1;
}
