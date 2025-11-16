// Function: sub_30B5770
// Address: 0x30b5770
//
void __fastcall sub_30B5770(__int64 a1, char a2)
{
  __int64 v2; // r15
  __int64 v5; // rdx
  __int64 *v6; // rax
  char v7; // al
  __m128i **v8; // rdx
  char v9; // al
  __m128i *v10; // rcx
  char v11; // dl
  void **v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rax
  _DWORD *v17; // rdx
  __int64 (__fastcall **v18)(); // rax
  __m128i *v19; // rsi
  __int64 v20; // rdx
  __int64 *v21; // rbx
  __int64 *v22; // r12
  unsigned __int64 v23; // r13
  __int64 v24; // rdi
  _WORD *v25; // rdx
  _QWORD *v26; // rdi
  _BYTE *v27; // rax
  _QWORD *v28; // rax
  __m128i *v29; // rdx
  __m128i si128; // xmm0
  __m128i v31; // xmm2
  void *v32; // [rsp+0h] [rbp-180h]
  __int64 v33; // [rsp+8h] [rbp-178h]
  __int64 v34; // [rsp+18h] [rbp-168h] BYREF
  unsigned __int8 *v35; // [rsp+20h] [rbp-160h] BYREF
  size_t v36; // [rsp+28h] [rbp-158h]
  __int64 v37; // [rsp+30h] [rbp-150h] BYREF
  _QWORD *v38; // [rsp+40h] [rbp-140h] BYREF
  __int64 (__fastcall **v39)(); // [rsp+48h] [rbp-138h]
  _QWORD v40[2]; // [rsp+50h] [rbp-130h] BYREF
  __m128i *v41; // [rsp+60h] [rbp-120h] BYREF
  __int64 *v42; // [rsp+68h] [rbp-118h]
  char v43; // [rsp+70h] [rbp-110h]
  char v44[15]; // [rsp+71h] [rbp-10Fh] BYREF
  __int16 v45; // [rsp+80h] [rbp-100h]
  __m128i v46; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v47; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+B0h] [rbp-D0h]
  void *v49[4]; // [rsp+C0h] [rbp-C0h] BYREF
  __int16 v50; // [rsp+E0h] [rbp-A0h]
  __m128i v51; // [rsp+F0h] [rbp-90h] BYREF
  __m128i v52; // [rsp+100h] [rbp-80h]
  __int64 v53; // [rsp+110h] [rbp-70h]

  v5 = *(_QWORD *)(a1 + 8);
  v49[0] = ".dot";
  v6 = *(__int64 **)(a1 + 16);
  v41 = (__m128i *)v5;
  v42 = v6;
  v50 = 259;
  v45 = 261;
  v38 = v40;
  sub_30B3180((__int64 *)&v38, (_BYTE *)qword_502EC48, qword_502EC48 + qword_502EC50);
  if ( v39 == (__int64 (__fastcall **)())0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v38, ".", 1u);
  v7 = v45;
  if ( !(_BYTE)v45 )
  {
    LOWORD(v48) = 256;
    goto LABEL_13;
  }
  if ( (_BYTE)v45 == 1 )
  {
    v9 = v50;
    v46.m128i_i64[0] = (__int64)&v38;
    LOWORD(v48) = 260;
    if ( (_BYTE)v50 )
    {
      if ( (_BYTE)v50 != 1 )
      {
        v2 = v46.m128i_i64[1];
        v10 = (__m128i *)&v38;
        v11 = 4;
        goto LABEL_9;
      }
LABEL_43:
      v31 = _mm_loadu_si128(&v47);
      v51 = _mm_loadu_si128(&v46);
      v53 = v48;
      v52 = v31;
      goto LABEL_14;
    }
LABEL_13:
    LOWORD(v53) = 256;
    goto LABEL_14;
  }
  if ( HIBYTE(v45) == 1 )
  {
    v8 = (__m128i **)v41;
    v33 = (__int64)v42;
  }
  else
  {
    v8 = &v41;
    v7 = 2;
  }
  BYTE1(v48) = v7;
  v9 = v50;
  v46.m128i_i64[0] = (__int64)&v38;
  v47.m128i_i64[0] = (__int64)v8;
  v47.m128i_i64[1] = v33;
  LOBYTE(v48) = 4;
  if ( !(_BYTE)v50 )
    goto LABEL_13;
  if ( (_BYTE)v50 == 1 )
    goto LABEL_43;
  v10 = &v46;
  v11 = 2;
LABEL_9:
  if ( HIBYTE(v50) == 1 )
  {
    v32 = v49[1];
    v12 = (void **)v49[0];
  }
  else
  {
    v12 = v49;
    v9 = 2;
  }
  v51.m128i_i64[0] = (__int64)v10;
  v51.m128i_i64[1] = v2;
  v52.m128i_i64[0] = (__int64)v12;
  v52.m128i_i64[1] = (__int64)v32;
  LOBYTE(v53) = v11;
  BYTE1(v53) = v9;
LABEL_14:
  sub_CA0F50((__int64 *)&v35, (void **)&v51);
  if ( v38 != v40 )
    j_j___libc_free_0((unsigned __int64)v38);
  v13 = sub_CB72A0();
  v14 = v13[4];
  v15 = (__int64)v13;
  if ( (unsigned __int64)(v13[3] - v14) <= 8 )
  {
    v15 = sub_CB6200((__int64)v13, "Writing '", 9u);
  }
  else
  {
    *(_BYTE *)(v14 + 8) = 39;
    *(_QWORD *)v14 = 0x20676E6974697257LL;
    v13[4] += 9LL;
  }
  v16 = sub_CB6200(v15, v35, v36);
  v17 = *(_DWORD **)(v16 + 32);
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 3u )
  {
    sub_CB6200(v16, "'...", 4u);
  }
  else
  {
    *v17 = 774778407;
    *(_QWORD *)(v16 + 32) += 4LL;
  }
  LODWORD(v38) = 0;
  v18 = sub_2241E40();
  v19 = (__m128i *)v35;
  v39 = v18;
  sub_CB7060((__int64)&v51, v35, v36, (__int64)&v38, 1u);
  if ( (_DWORD)v38 )
  {
    v28 = sub_CB72A0();
    v29 = (__m128i *)v28[4];
    if ( v28[3] - (_QWORD)v29 <= 0x20u )
    {
      v19 = (__m128i *)"  error opening file for writing!";
      sub_CB6200((__int64)v28, "  error opening file for writing!", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F95580);
      v29[2].m128i_i8[0] = 33;
      *v29 = si128;
      v29[1] = _mm_load_si128((const __m128i *)&xmmword_3F95590);
      v28[4] += 33LL;
    }
  }
  else
  {
    v34 = a1;
    v50 = 257;
    v42 = &v34;
    v41 = &v51;
    v43 = 0;
    v44[0] = a2;
    sub_CA0F50(v46.m128i_i64, v49);
    v19 = &v46;
    sub_30B42E0((__int64 *)&v41, (__int64)&v46);
    v20 = *v42;
    v21 = *(__int64 **)(*v42 + 96);
    v22 = &v21[*(unsigned int *)(*v42 + 104)];
    if ( v21 != v22 )
    {
      while ( 1 )
      {
        v23 = *v21;
        v19 = (__m128i *)*v21;
        if ( sub_30B3230(v44, *v21, v20) )
        {
          if ( v22 == ++v21 )
            break;
        }
        else
        {
          v19 = (__m128i *)v23;
          ++v21;
          sub_30B4ED0((__int64)&v41, v23);
          if ( v22 == v21 )
            break;
        }
        v20 = *v42;
      }
    }
    v24 = (__int64)v41;
    v25 = (_WORD *)v41[2].m128i_i64[0];
    if ( v41[1].m128i_i64[1] - (__int64)v25 <= 1uLL )
    {
      v19 = (__m128i *)"}\n";
      sub_CB6200((__int64)v41, "}\n", 2u);
    }
    else
    {
      *v25 = 2685;
      *(_QWORD *)(v24 + 32) += 2LL;
    }
    if ( (__m128i *)v46.m128i_i64[0] != &v47 )
    {
      v19 = (__m128i *)(v47.m128i_i64[0] + 1);
      j_j___libc_free_0(v46.m128i_u64[0]);
    }
  }
  v26 = sub_CB72A0();
  v27 = (_BYTE *)v26[4];
  if ( (_BYTE *)v26[3] == v27 )
  {
    v19 = (__m128i *)"\n";
    sub_CB6200((__int64)v26, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v27 = 10;
    ++v26[4];
  }
  sub_CB5B00(v51.m128i_i32, (__int64)v19);
  if ( v35 != (unsigned __int8 *)&v37 )
    j_j___libc_free_0((unsigned __int64)v35);
}
