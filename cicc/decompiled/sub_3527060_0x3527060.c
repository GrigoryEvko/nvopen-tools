// Function: sub_3527060
// Address: 0x3527060
//
void __fastcall sub_3527060(__int64 *a1)
{
  __int64 v1; // r14
  __int64 ***v3; // rax
  const void *v4; // r15
  size_t v5; // r13
  __int64 ***v6; // rdx
  _QWORD *v7; // rax
  char v8; // al
  __int64 ****v9; // rdx
  char v10; // al
  __m128i *v11; // rcx
  char v12; // dl
  void **v13; // rsi
  __m128i v14; // xmm2
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  _DWORD *v19; // rdx
  __int64 (__fastcall **v20)(); // rax
  __int64 v21; // rsi
  __int64 *v22; // rax
  unsigned __int64 v23; // rbx
  __int64 *i; // r12
  __int64 ***v25; // rdi
  __int64 **v26; // rdx
  _QWORD *v27; // rdi
  _BYTE *v28; // rax
  _QWORD *v29; // rax
  __m128i *v30; // rdx
  __m128i si128; // xmm0
  _QWORD *v32; // rdi
  void *v33; // [rsp+0h] [rbp-180h]
  __int64 v34; // [rsp+8h] [rbp-178h]
  __int64 *v35; // [rsp+10h] [rbp-170h] BYREF
  __int64 **v36; // [rsp+18h] [rbp-168h] BYREF
  unsigned __int8 *v37; // [rsp+20h] [rbp-160h] BYREF
  size_t v38; // [rsp+28h] [rbp-158h]
  __int64 v39; // [rsp+30h] [rbp-150h] BYREF
  _QWORD *v40; // [rsp+40h] [rbp-140h] BYREF
  size_t v41; // [rsp+48h] [rbp-138h]
  _QWORD v42[2]; // [rsp+50h] [rbp-130h] BYREF
  __int64 ****v43; // [rsp+60h] [rbp-120h] BYREF
  __int64 ***v44; // [rsp+68h] [rbp-118h]
  char v45; // [rsp+70h] [rbp-110h]
  char v46; // [rsp+71h] [rbp-10Fh]
  __int16 v47; // [rsp+80h] [rbp-100h]
  __m128i v48; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v49; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v50; // [rsp+B0h] [rbp-D0h]
  void *v51[4]; // [rsp+C0h] [rbp-C0h] BYREF
  __int16 v52; // [rsp+E0h] [rbp-A0h]
  __m128i v53; // [rsp+F0h] [rbp-90h] BYREF
  __m128i v54; // [rsp+100h] [rbp-80h]
  __int64 v55; // [rsp+110h] [rbp-70h]

  v51[0] = ".dot";
  v52 = 259;
  v3 = (__int64 ***)sub_2E791E0(a1);
  v4 = qword_503D0C8;
  v5 = qword_503D0D0;
  v43 = (__int64 ****)v3;
  v47 = 261;
  v44 = v6;
  v40 = v42;
  if ( (char *)qword_503D0C8 + qword_503D0D0 && !qword_503D0C8 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v53.m128i_i64[0] = qword_503D0D0;
  if ( qword_503D0D0 > 0xF )
  {
    v40 = (_QWORD *)sub_22409D0((__int64)&v40, (unsigned __int64 *)&v53, 0);
    v32 = v40;
    v42[0] = v53.m128i_i64[0];
  }
  else
  {
    if ( qword_503D0D0 == 1 )
    {
      LOBYTE(v42[0]) = *(_BYTE *)qword_503D0C8;
      v7 = v42;
      goto LABEL_6;
    }
    if ( !qword_503D0D0 )
    {
      v7 = v42;
      goto LABEL_6;
    }
    v32 = v42;
  }
  memcpy(v32, v4, v5);
  v5 = v53.m128i_i64[0];
  v7 = v40;
LABEL_6:
  v41 = v5;
  *((_BYTE *)v7 + v5) = 0;
  if ( v41 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v40, ".", 1u);
  v8 = v47;
  if ( !(_BYTE)v47 )
  {
    LOWORD(v50) = 256;
    goto LABEL_23;
  }
  if ( (_BYTE)v47 == 1 )
  {
    v10 = v52;
    v48.m128i_i64[0] = (__int64)&v40;
    LOWORD(v50) = 260;
    if ( (_BYTE)v52 )
    {
      if ( (_BYTE)v52 != 1 )
      {
        v1 = v48.m128i_i64[1];
        v11 = (__m128i *)&v40;
        v12 = 4;
        goto LABEL_14;
      }
LABEL_21:
      v14 = _mm_loadu_si128(&v49);
      v53 = _mm_loadu_si128(&v48);
      v55 = v50;
      v54 = v14;
      goto LABEL_24;
    }
LABEL_23:
    LOWORD(v55) = 256;
    goto LABEL_24;
  }
  if ( HIBYTE(v47) == 1 )
  {
    v9 = v43;
    v34 = (__int64)v44;
  }
  else
  {
    v9 = (__int64 ****)&v43;
    v8 = 2;
  }
  BYTE1(v50) = v8;
  v10 = v52;
  v48.m128i_i64[0] = (__int64)&v40;
  v49.m128i_i64[0] = (__int64)v9;
  v49.m128i_i64[1] = v34;
  LOBYTE(v50) = 4;
  if ( !(_BYTE)v52 )
    goto LABEL_23;
  if ( (_BYTE)v52 == 1 )
    goto LABEL_21;
  v11 = &v48;
  v12 = 2;
LABEL_14:
  if ( HIBYTE(v52) == 1 )
  {
    v33 = v51[1];
    v13 = (void **)v51[0];
  }
  else
  {
    v13 = v51;
    v10 = 2;
  }
  v53.m128i_i64[0] = (__int64)v11;
  v53.m128i_i64[1] = v1;
  v54.m128i_i64[0] = (__int64)v13;
  v54.m128i_i64[1] = (__int64)v33;
  LOBYTE(v55) = v12;
  BYTE1(v55) = v10;
LABEL_24:
  sub_CA0F50((__int64 *)&v37, (void **)&v53);
  if ( v40 != v42 )
    j_j___libc_free_0((unsigned __int64)v40);
  v15 = sub_CB72A0();
  v16 = v15[4];
  v17 = (__int64)v15;
  if ( (unsigned __int64)(v15[3] - v16) <= 8 )
  {
    v17 = sub_CB6200((__int64)v15, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v16 + 8) = 39;
    *(_QWORD *)v16 = 0x20676E6974697257LL;
    v15[4] += 9LL;
  }
  v18 = sub_CB6200(v17, v37, v38);
  v19 = *(_DWORD **)(v18 + 32);
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 3u )
  {
    sub_CB6200(v18, "'...", 4u);
  }
  else
  {
    *v19 = 774778407;
    *(_QWORD *)(v18 + 32) += 4LL;
  }
  LODWORD(v40) = 0;
  v20 = sub_2241E40();
  v21 = (__int64)v37;
  v41 = (size_t)v20;
  sub_CB7060((__int64)&v53, v37, v38, (__int64)&v40, 1u);
  v35 = a1;
  if ( (_DWORD)v40 )
  {
    v29 = sub_CB72A0();
    v30 = (__m128i *)v29[4];
    if ( v29[3] - (_QWORD)v30 <= 0x20u )
    {
      v21 = (__int64)"  error opening file for writing!";
      sub_CB6200((__int64)v29, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v30[2].m128i_i8[0] = 33;
      *v30 = si128;
      v30[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v29[4] += 33LL;
    }
  }
  else
  {
    v43 = (__int64 ****)&v53;
    v36 = &v35;
    v44 = &v36;
    v52 = 257;
    v46 = qword_503CFE8;
    v45 = 0;
    sub_CA0F50(v48.m128i_i64, v51);
    v21 = (__int64)&v48;
    sub_35258A0((__int64 ****)&v43, (__int64)&v48);
    v22 = **v44;
    v23 = v22[41];
    for ( i = v22 + 40; i != (__int64 *)v23; v23 = *(_QWORD *)(v23 + 8) )
    {
      v21 = v23;
      sub_3526200(&v43, v23);
    }
    v25 = (__int64 ***)v43;
    v26 = (__int64 **)v43[4];
    if ( (unsigned __int64)((char *)v43[3] - (char *)v26) <= 1 )
    {
      v21 = (__int64)"}\n";
      sub_CB6200((__int64)v43, "}\n", 2u);
    }
    else
    {
      *(_WORD *)v26 = 2685;
      v25[4] = (__int64 **)((char *)v25[4] + 2);
    }
    if ( (__m128i *)v48.m128i_i64[0] != &v49 )
    {
      v21 = v49.m128i_i64[0] + 1;
      j_j___libc_free_0(v48.m128i_u64[0]);
    }
  }
  v27 = sub_CB72A0();
  v28 = (_BYTE *)v27[4];
  if ( (unsigned __int64)v28 >= v27[3] )
  {
    v21 = 10;
    sub_CB5D20((__int64)v27, 10);
  }
  else
  {
    v27[4] = v28 + 1;
    *v28 = 10;
  }
  sub_CB5B00(v53.m128i_i32, v21);
  if ( v37 != (unsigned __int8 *)&v39 )
    j_j___libc_free_0((unsigned __int64)v37);
}
