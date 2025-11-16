// Function: sub_273A870
// Address: 0x273a870
//
void __fastcall sub_273A870(__m128i *src, const __m128i *a2)
{
  const __m128i *v2; // r15
  unsigned __int32 v3; // r13d
  __int32 v4; // eax
  __int32 v5; // r12d
  __m128i *v6; // rbx
  __m128i v7; // xmm4
  __m128i v8; // xmm5
  _BYTE *v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm4
  __m128i v12; // xmm5
  __m128i v13; // xmm6
  __int32 v14; // esi
  bool v15; // al
  __int64 v16; // rax
  __m128i v17; // xmm7
  __m128i v18; // xmm7
  __int32 v19; // esi
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __int64 v23; // rax
  __m128i v24; // xmm7
  __m128i v25; // xmm7
  unsigned __int64 v26; // r10
  unsigned __int64 v27; // r11
  unsigned __int64 v28; // r9
  __int64 v29; // r10
  bool v30; // dl
  _BYTE *v31; // [rsp+0h] [rbp-90h]
  _BYTE *v32; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  __m128i v35; // [rsp+20h] [rbp-70h] BYREF
  __m128i v36; // [rsp+30h] [rbp-60h] BYREF
  __m128i v37; // [rsp+40h] [rbp-50h] BYREF
  __m128i v38; // [rsp+50h] [rbp-40h]

  if ( src != a2 && a2 != &src[4] )
  {
    v2 = src + 4;
    do
    {
      while ( 1 )
      {
        v3 = v2[3].m128i_u32[0];
        if ( v3 != src[3].m128i_i32[0] )
          break;
        v4 = v2[3].m128i_i32[2];
        v19 = src[3].m128i_i32[2];
        v5 = v4;
        if ( v4 )
        {
          if ( !v19 )
            goto LABEL_40;
          v28 = v2->m128i_i64[0];
          if ( v4 == 3 )
            v28 = sub_2739680(v2->m128i_i64[0]);
          v29 = src->m128i_i64[0];
          if ( v19 == 3 )
            v29 = sub_2739680(src->m128i_i64[0]);
          if ( !sub_B445A0(v28, v29) )
          {
            v3 = v2[3].m128i_u32[0];
            v5 = v2[3].m128i_i32[2];
LABEL_40:
            v32 = (_BYTE *)v2->m128i_i64[1];
            goto LABEL_8;
          }
        }
        else if ( !v19 )
        {
          v30 = 0;
          v32 = (_BYTE *)v2->m128i_i64[1];
          if ( *v32 != 17 )
            v30 = *(_BYTE *)v2[1].m128i_i64[0] != 17;
          if ( *(_BYTE *)src->m128i_i64[1] == 17
            || (unsigned __int8)v30 >= (unsigned __int8)(*(_BYTE *)src[1].m128i_i64[0] != 17) )
          {
            goto LABEL_7;
          }
        }
LABEL_22:
        v20 = _mm_loadu_si128(v2 + 1);
        v21 = _mm_loadu_si128(v2 + 2);
        v22 = _mm_loadu_si128(v2 + 3);
        v35 = _mm_loadu_si128(v2);
        v36 = v20;
        v37 = v21;
        v38 = v22;
        if ( src != v2 )
          memmove(&src[4], src, (char *)v2 - (char *)src);
        v23 = v38.m128i_i64[0];
        v2 += 4;
        *src = _mm_loadu_si128(&v35);
        v24 = _mm_loadu_si128(&v36);
        src[3].m128i_i64[0] = v23;
        LODWORD(v23) = v38.m128i_i32[2];
        src[1] = v24;
        v25 = _mm_loadu_si128(&v37);
        src[3].m128i_i32[2] = v23;
        src[2] = v25;
        if ( a2 == v2 )
          return;
      }
      if ( v3 < src[3].m128i_i32[0] )
        goto LABEL_22;
      v4 = v2[3].m128i_i32[2];
      v32 = (_BYTE *)v2->m128i_i64[1];
LABEL_7:
      v5 = v4;
LABEL_8:
      v6 = (__m128i *)v2;
      v7 = _mm_loadu_si128(v2 + 2);
      v8 = _mm_loadu_si128(v2 + 3);
      v34 = v2->m128i_i64[0];
      v9 = (_BYTE *)v2[1].m128i_i64[0];
      v35 = _mm_loadu_si128(v2);
      v31 = v9;
      v36 = _mm_loadu_si128(v2 + 1);
      v37 = v7;
      v38 = v8;
      while ( 1 )
      {
        if ( v6[-1].m128i_i32[0] != v3 )
        {
          if ( v6[-1].m128i_i32[0] <= v3 )
            goto LABEL_18;
          goto LABEL_10;
        }
        v14 = v6[-1].m128i_i32[2];
        if ( v5 )
          break;
        if ( !v14 )
        {
          v15 = 0;
          if ( *v32 != 17 )
            v15 = *v31 != 17;
          if ( *(_BYTE *)v6[-4].m128i_i64[1] == 17
            || (unsigned __int8)v15 >= (unsigned __int8)(*(_BYTE *)v6[-3].m128i_i64[0] != 17) )
          {
            goto LABEL_18;
          }
        }
LABEL_10:
        v10 = v6[-1].m128i_i64[0];
        v11 = _mm_loadu_si128(v6 - 4);
        v6 -= 4;
        v12 = _mm_loadu_si128(v6 + 1);
        v13 = _mm_loadu_si128(v6 + 2);
        v6[7].m128i_i64[0] = v10;
        LODWORD(v10) = v6[3].m128i_i32[2];
        v6[4] = v11;
        v6[7].m128i_i32[2] = v10;
        v6[5] = v12;
        v6[6] = v13;
      }
      if ( v14 )
      {
        v26 = v34;
        if ( v5 == 3 )
          v26 = sub_2739680(v34);
        v27 = v6[-4].m128i_u64[0];
        if ( v14 == 3 )
          v27 = sub_2739680(v6[-4].m128i_i64[0]);
        if ( sub_B445A0(v26, v27) )
          goto LABEL_10;
      }
LABEL_18:
      v16 = v38.m128i_i64[0];
      v2 += 4;
      *v6 = _mm_loadu_si128(&v35);
      v17 = _mm_loadu_si128(&v36);
      v6[3].m128i_i64[0] = v16;
      LODWORD(v16) = v38.m128i_i32[2];
      v6[1] = v17;
      v18 = _mm_loadu_si128(&v37);
      v6[3].m128i_i32[2] = v16;
      v6[2] = v18;
    }
    while ( a2 != v2 );
  }
}
