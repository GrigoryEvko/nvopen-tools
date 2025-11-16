// Function: sub_A89620
// Address: 0xa89620
//
_QWORD *__fastcall sub_A89620(_QWORD *a1)
{
  _QWORD *result; // rax
  _QWORD *v3; // r14
  unsigned __int8 v4; // dl
  __int64 v5; // rdi
  __int64 v6; // rax
  size_t v7; // rdx
  _QWORD *v8; // rcx
  __int64 v9; // r11
  unsigned int v10; // edx
  const char *v11; // rsi
  _QWORD *v12; // r12
  __int64 v13; // rax
  _BYTE *v14; // r9
  size_t v15; // r8
  _QWORD *v16; // rax
  _BYTE *v17; // r9
  size_t v18; // r8
  _QWORD *v19; // rax
  __m128i *v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r11
  __m128i *v23; // rax
  __m128i *v24; // rcx
  __m128i *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rdi
  size_t n; // [rsp+0h] [rbp-2C0h]
  _BYTE *src; // [rsp+8h] [rbp-2B8h]
  size_t v33; // [rsp+10h] [rbp-2B0h]
  _BYTE *v34; // [rsp+18h] [rbp-2A8h]
  __int64 v35; // [rsp+20h] [rbp-2A0h]
  __int64 v36; // [rsp+20h] [rbp-2A0h]
  _QWORD *v37; // [rsp+38h] [rbp-288h] BYREF
  __m128i *v38; // [rsp+40h] [rbp-280h]
  __int64 v39; // [rsp+48h] [rbp-278h]
  __m128i v40; // [rsp+50h] [rbp-270h] BYREF
  _QWORD *v41; // [rsp+60h] [rbp-260h] BYREF
  size_t v42; // [rsp+68h] [rbp-258h]
  _QWORD v43[2]; // [rsp+70h] [rbp-250h] BYREF
  __m128i *v44; // [rsp+80h] [rbp-240h] BYREF
  __int64 v45; // [rsp+88h] [rbp-238h]
  __m128i v46; // [rsp+90h] [rbp-230h] BYREF
  _QWORD *v47; // [rsp+A0h] [rbp-220h] BYREF
  size_t v48; // [rsp+A8h] [rbp-218h]
  _QWORD v49[2]; // [rsp+B0h] [rbp-210h] BYREF
  _QWORD v50[58]; // [rsp+C0h] [rbp-200h] BYREF
  _BYTE v51[48]; // [rsp+290h] [rbp-30h] BYREF

  v37 = a1;
  sub_A89080(&v37, "clang.arc.use", 0x104u);
  result = (_QWORD *)sub_BA8DC0(a1, "clang.arc.retainAutoreleasedReturnValueMarker", 45);
  if ( !result )
    return result;
  v3 = result;
  result = (_QWORD *)sub_B91A10(result, 0);
  if ( !result )
    return result;
  v4 = *((_BYTE *)result - 16);
  result = (v4 & 2) != 0 ? (_QWORD *)*(result - 4) : &result[-((v4 >> 2) & 0xF) - 2];
  if ( !*result || *(_BYTE *)*result )
    return result;
  v5 = *result;
  v35 = *result;
  v50[0] = &v50[2];
  v50[1] = 0x400000000LL;
  v6 = sub_B91420(v5, 0);
  v48 = v7;
  v47 = (_QWORD *)v6;
  sub_C937F0(&v47, v50, "#", 1, 0xFFFFFFFFLL, 1);
  v9 = v35;
  if ( LODWORD(v50[1]) == 2 )
  {
    v13 = v50[0];
    v14 = *(_BYTE **)(v50[0] + 16LL);
    if ( !v14 )
    {
      v8 = v49;
      LOBYTE(v49[0]) = 0;
      v47 = v49;
      v48 = 0;
LABEL_20:
      v17 = *(_BYTE **)v13;
      if ( !*(_QWORD *)v13 )
      {
        LOBYTE(v43[0]) = 0;
        v41 = v43;
        v42 = 0;
        goto LABEL_25;
      }
      v18 = *(_QWORD *)(v13 + 8);
      v41 = v43;
      v44 = (__m128i *)v18;
      if ( v18 > 0xF )
      {
        n = v18;
        src = v17;
        v27 = sub_22409D0(&v41, &v44, 0);
        v17 = src;
        v41 = (_QWORD *)v27;
        v28 = (_QWORD *)v27;
        v18 = n;
        v43[0] = v44;
      }
      else
      {
        if ( v18 == 1 )
        {
          LOBYTE(v43[0]) = *v17;
          v19 = v43;
          goto LABEL_24;
        }
        if ( !v18 )
        {
          v19 = v43;
LABEL_24:
          v42 = v18;
          *((_BYTE *)v19 + v18) = 0;
          if ( v42 == 0x3FFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"basic_string::append");
LABEL_25:
          v20 = (__m128i *)sub_2241490(&v41, ";", 1, v8);
          v44 = &v46;
          if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
          {
            v46 = _mm_loadu_si128(v20 + 1);
          }
          else
          {
            v44 = (__m128i *)v20->m128i_i64[0];
            v46.m128i_i64[0] = v20[1].m128i_i64[0];
          }
          v45 = v20->m128i_i64[1];
          v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
          v20->m128i_i64[1] = 0;
          v20[1].m128i_i8[0] = 0;
          v21 = 15;
          v22 = 15;
          if ( v44 != &v46 )
            v22 = v46.m128i_i64[0];
          if ( v45 + v48 <= v22 )
            goto LABEL_33;
          if ( v47 != v49 )
            v21 = v49[0];
          if ( v45 + v48 <= v21 )
          {
            v23 = (__m128i *)sub_2241130(&v47, 0, 0, v44, v45);
            v38 = &v40;
            v24 = (__m128i *)v23->m128i_i64[0];
            v25 = v23 + 1;
            if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
              goto LABEL_34;
          }
          else
          {
LABEL_33:
            v23 = (__m128i *)sub_2241490(&v44, v47, v48, v44);
            v38 = &v40;
            v24 = (__m128i *)v23->m128i_i64[0];
            v25 = v23 + 1;
            if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
            {
LABEL_34:
              v38 = v24;
              v40.m128i_i64[0] = v23[1].m128i_i64[0];
LABEL_35:
              v39 = v23->m128i_i64[1];
              v23->m128i_i64[0] = (__int64)v25;
              v23->m128i_i64[1] = 0;
              v23[1].m128i_i8[0] = 0;
              if ( v44 != &v46 )
                j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
              if ( v41 != v43 )
                j_j___libc_free_0(v41, v43[0] + 1LL);
              if ( v47 != v49 )
                j_j___libc_free_0(v47, v49[0] + 1LL);
              v26 = sub_B9B140(*a1, v38, v39);
              v9 = v26;
              if ( v38 != &v40 )
              {
                v36 = v26;
                j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
                v9 = v36;
              }
              goto LABEL_8;
            }
          }
          v40 = _mm_loadu_si128(v23 + 1);
          goto LABEL_35;
        }
        v28 = v43;
      }
      memcpy(v28, v17, v18);
      v18 = (size_t)v44;
      v19 = v41;
      goto LABEL_24;
    }
    v15 = *(_QWORD *)(v50[0] + 24LL);
    v47 = v49;
    v44 = (__m128i *)v15;
    if ( v15 > 0xF )
    {
      v33 = v15;
      v34 = v14;
      v29 = sub_22409D0(&v47, &v44, 0);
      v14 = v34;
      v15 = v33;
      v47 = (_QWORD *)v29;
      v30 = (_QWORD *)v29;
      v49[0] = v44;
    }
    else
    {
      if ( v15 == 1 )
      {
        LOBYTE(v49[0]) = *v14;
        v16 = v49;
LABEL_19:
        v48 = v15;
        *((_BYTE *)v16 + v15) = 0;
        v13 = v50[0];
        goto LABEL_20;
      }
      if ( !v15 )
      {
        v16 = v49;
        goto LABEL_19;
      }
      v30 = v49;
    }
    memcpy(v30, v14, v15);
    v15 = (size_t)v44;
    v16 = v47;
    goto LABEL_19;
  }
LABEL_8:
  sub_BA92F0(a1, 1, "clang.arc.retainAutoreleasedReturnValueMarker", 45, v9);
  sub_BA9050(a1, v3);
  if ( (_QWORD *)v50[0] != &v50[2] )
    _libc_free(v50[0], v3);
  v10 = 255;
  memset(v50, 0, sizeof(v50));
  v11 = "objc_autorelease";
  LODWORD(v50[1]) = 255;
  v50[2] = "objc_autoreleasePoolPop";
  v12 = v50;
  v50[4] = "objc_autoreleasePoolPush";
  v50[6] = "objc_autoreleaseReturnValue";
  v50[8] = "objc_copyWeak";
  v50[10] = "objc_destroyWeak";
  v50[12] = "objc_initWeak";
  v50[14] = "objc_loadWeak";
  v50[16] = "objc_loadWeakRetained";
  v50[18] = "objc_moveWeak";
  v50[20] = "objc_release";
  v50[22] = "objc_retain";
  v50[24] = "objc_retainAutorelease";
  v50[26] = "objc_retainAutoreleaseReturnValue";
  v50[28] = "objc_retainAutoreleasedReturnValue";
  v50[30] = "objc_retainBlock";
  v50[0] = "objc_autorelease";
  LODWORD(v50[3]) = 256;
  LODWORD(v50[5]) = 257;
  LODWORD(v50[7]) = 258;
  LODWORD(v50[9]) = 261;
  LODWORD(v50[11]) = 262;
  LODWORD(v50[13]) = 263;
  LODWORD(v50[15]) = 264;
  LODWORD(v50[17]) = 265;
  LODWORD(v50[19]) = 266;
  LODWORD(v50[21]) = 267;
  LODWORD(v50[23]) = 268;
  LODWORD(v50[25]) = 270;
  LODWORD(v50[27]) = 271;
  LODWORD(v50[29]) = 272;
  LODWORD(v50[31]) = 273;
  v50[32] = "objc_storeStrong";
  v50[34] = "objc_storeWeak";
  v50[36] = "objc_unsafeClaimAutoreleasedReturnValue";
  v50[38] = "objc_retainedObject";
  v50[40] = "objc_unretainedObject";
  v50[42] = "objc_unretainedPointer";
  v50[44] = "objc_retain_autorelease";
  v50[46] = "objc_sync_enter";
  v50[48] = "objc_sync_exit";
  v50[50] = "objc_arc_annotation_topdown_bbstart";
  v50[52] = "objc_arc_annotation_topdown_bbend";
  v50[54] = "objc_arc_annotation_bottomup_bbstart";
  LODWORD(v50[33]) = 275;
  LODWORD(v50[35]) = 276;
  LODWORD(v50[37]) = 281;
  LODWORD(v50[39]) = 274;
  LODWORD(v50[41]) = 279;
  LODWORD(v50[43]) = 280;
  LODWORD(v50[45]) = 269;
  LODWORD(v50[47]) = 277;
  LODWORD(v50[49]) = 278;
  LODWORD(v50[51]) = 254;
  LODWORD(v50[53]) = 253;
  LODWORD(v50[55]) = 252;
  v50[56] = "objc_arc_annotation_bottomup_bbend";
  LODWORD(v50[57]) = 251;
  while ( 1 )
  {
    v12 += 2;
    sub_A89080(&v37, v11, v10);
    result = v51;
    if ( v12 == (_QWORD *)v51 )
      break;
    v10 = *((_DWORD *)v12 + 2);
    v11 = (const char *)*v12;
  }
  return result;
}
