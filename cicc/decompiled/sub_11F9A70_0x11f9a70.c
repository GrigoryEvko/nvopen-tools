// Function: sub_11F9A70
// Address: 0x11f9a70
//
_QWORD *__fastcall sub_11F9A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // r13
  __int64 v6; // r15
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  char v11; // al
  _QWORD *v12; // rdx
  char v13; // al
  __int64 **v14; // rcx
  char v15; // dl
  _QWORD *v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  const char *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  _DWORD *v24; // rdx
  __int64 v25; // rdi
  char *v26; // rsi
  _QWORD *v27; // rdi
  _BYTE *v28; // rax
  _QWORD *result; // rax
  _QWORD *v30; // rax
  __m128i *v31; // rdx
  __m128i si128; // xmm0
  __m128i v33; // xmm2
  __int64 v34; // [rsp+0h] [rbp-190h]
  unsigned __int8 *v39; // [rsp+30h] [rbp-160h] BYREF
  size_t v40; // [rsp+38h] [rbp-158h]
  _QWORD v41[2]; // [rsp+40h] [rbp-150h] BYREF
  __int64 *v42[2]; // [rsp+50h] [rbp-140h] BYREF
  _QWORD v43[2]; // [rsp+60h] [rbp-130h] BYREF
  const char *v44; // [rsp+70h] [rbp-120h] BYREF
  __int64 v45; // [rsp+78h] [rbp-118h]
  __int16 v46; // [rsp+90h] [rbp-100h]
  __m128i v47; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v48; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v49; // [rsp+C0h] [rbp-D0h]
  _QWORD v50[4]; // [rsp+D0h] [rbp-C0h] BYREF
  char v51; // [rsp+F0h] [rbp-A0h]
  char v52; // [rsp+F1h] [rbp-9Fh]
  char v53; // [rsp+F8h] [rbp-98h]
  char v54; // [rsp+F9h] [rbp-97h]
  char v55; // [rsp+FAh] [rbp-96h]
  __m128i v56; // [rsp+100h] [rbp-90h] BYREF
  __m128i v57; // [rsp+110h] [rbp-80h]
  __int64 v58; // [rsp+120h] [rbp-70h]

  v52 = 1;
  v50[0] = ".dot";
  v51 = 3;
  v8 = sub_BD5D20(a1);
  v45 = v9;
  v44 = v8;
  v46 = 261;
  v42[0] = v43;
  sub_11F4570((__int64 *)v42, (_BYTE *)qword_4F92208, qword_4F92208 + qword_4F92210);
  if ( v42[1] == (__int64 *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(v42, ".", 1, v10);
  v11 = v46;
  if ( !(_BYTE)v46 )
  {
    LOWORD(v49) = 256;
    goto LABEL_13;
  }
  if ( (_BYTE)v46 == 1 )
  {
    v13 = v51;
    v47.m128i_i64[0] = (__int64)v42;
    LOWORD(v49) = 260;
    if ( v51 )
    {
      if ( v51 != 1 )
      {
        v5 = v47.m128i_i64[1];
        v14 = v42;
        v15 = 4;
        goto LABEL_9;
      }
LABEL_34:
      v33 = _mm_loadu_si128(&v48);
      v56 = _mm_loadu_si128(&v47);
      v58 = v49;
      v57 = v33;
      goto LABEL_14;
    }
LABEL_13:
    LOWORD(v58) = 256;
    goto LABEL_14;
  }
  if ( HIBYTE(v46) == 1 )
  {
    v6 = v45;
    v12 = v44;
  }
  else
  {
    v12 = &v44;
    v11 = 2;
  }
  BYTE1(v49) = v11;
  v13 = v51;
  v47.m128i_i64[0] = (__int64)v42;
  v48.m128i_i64[0] = (__int64)v12;
  v48.m128i_i64[1] = v6;
  LOBYTE(v49) = 4;
  if ( !v51 )
    goto LABEL_13;
  if ( v51 == 1 )
    goto LABEL_34;
  v14 = (__int64 **)&v47;
  v15 = 2;
LABEL_9:
  if ( v52 == 1 )
  {
    v34 = v50[1];
    v16 = (_QWORD *)v50[0];
  }
  else
  {
    v13 = 2;
    v16 = v50;
  }
  v56.m128i_i64[0] = (__int64)v14;
  v56.m128i_i64[1] = v5;
  v57.m128i_i64[0] = (__int64)v16;
  v57.m128i_i64[1] = v34;
  LOBYTE(v58) = v15;
  BYTE1(v58) = v13;
LABEL_14:
  sub_CA0F50((__int64 *)&v39, (void **)&v56);
  if ( v42[0] != v43 )
    j_j___libc_free_0(v42[0], v43[0] + 1LL);
  v17 = sub_CB72A0();
  v18 = v17[4];
  v19 = (__int64)v17;
  if ( (unsigned __int64)(v17[3] - v18) <= 8 )
  {
    v19 = sub_CB6200((__int64)v17, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v18 + 8) = 39;
    *(_QWORD *)v18 = 0x20676E6974697257LL;
    v17[4] += 9LL;
  }
  v20 = (const char *)v39;
  v21 = sub_CB6200(v19, v39, v40);
  v24 = *(_DWORD **)(v21 + 32);
  v25 = v21;
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v24 <= 3u )
  {
    v20 = "'...";
    sub_CB6200(v21, "'...", 4u);
  }
  else
  {
    *v24 = 774778407;
    *(_QWORD *)(v21 + 32) += 4LL;
  }
  LODWORD(v44) = 0;
  v45 = sub_2241E40(v25, v20, v24, v22, v23);
  sub_CB7060((__int64)&v56, v39, v40, (__int64)&v44, 1u);
  v26 = (char *)a1;
  sub_11F3840((__int64)v50, a1, a2, a3, a4);
  v53 = byte_4F91E88;
  v54 = qword_4F91CC8;
  v55 = byte_4F91DA8;
  if ( (_DWORD)v44 )
  {
    v30 = sub_CB72A0();
    v31 = (__m128i *)v30[4];
    if ( v30[3] - (_QWORD)v31 <= 0x20u )
    {
      v26 = "  error opening file for writing!";
      sub_CB6200((__int64)v30, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v31[2].m128i_i8[0] = 33;
      *v31 = si128;
      v31[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v30[4] += 33LL;
    }
  }
  else
  {
    v26 = (char *)v42;
    LOWORD(v49) = 257;
    v42[0] = v50;
    sub_11F98E0((__int64 **)&v56, v42, a5, (void **)&v47);
  }
  v27 = sub_CB72A0();
  v28 = (_BYTE *)v27[4];
  if ( (_BYTE *)v27[3] == v28 )
  {
    v26 = "\n";
    sub_CB6200((__int64)v27, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v28 = 10;
    ++v27[4];
  }
  sub_11F3870((__int64)v50);
  sub_CB5B00(v56.m128i_i32, (__int64)v26);
  result = v41;
  if ( v39 != (unsigned __int8 *)v41 )
    return (_QWORD *)j_j___libc_free_0(v39, v41[0] + 1LL);
  return result;
}
