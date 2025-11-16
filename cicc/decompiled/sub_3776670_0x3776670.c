// Function: sub_3776670
// Address: 0x3776670
//
__m128i *__fastcall sub_3776670(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8)
{
  unsigned __int16 v8; // r12
  bool v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // al
  __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned int v17; // eax
  __m128i v18; // xmm0
  __int64 *v19; // r12
  __int64 i; // r14
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int16 v24; // ax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __m128i v28; // xmm4
  __m128i v29; // xmm1
  __int64 v31; // r15
  _BYTE *v32; // r14
  __int64 v33; // rax
  char v34; // dl
  unsigned int v35; // ebx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // ecx
  __m128i v39; // xmm2
  unsigned int v40; // [rsp+Ch] [rbp-F4h]
  unsigned int v41; // [rsp+28h] [rbp-D8h]
  unsigned int v42; // [rsp+2Ch] [rbp-D4h]
  unsigned int v43; // [rsp+30h] [rbp-D0h]
  __int64 v44; // [rsp+30h] [rbp-D0h]
  __int64 v46; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-B8h]
  __m128i v48; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v50; // [rsp+70h] [rbp-90h] BYREF
  __int64 v51; // [rsp+80h] [rbp-80h]
  __int64 v52; // [rsp+88h] [rbp-78h]
  __int64 v53; // [rsp+90h] [rbp-70h]
  __int64 v54; // [rsp+98h] [rbp-68h]
  __int64 v55; // [rsp+A0h] [rbp-60h]
  __int64 v56; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v57; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v58; // [rsp+B8h] [rbp-48h]

  v41 = a4;
  v46 = a5;
  v47 = a6;
  if ( (_WORD)a5 )
  {
    v48.m128i_i64[1] = 0;
    v8 = word_4456580[(unsigned __int16)a5 - 1];
    v48.m128i_i16[0] = v8;
  }
  else
  {
    v13 = sub_3009970((__int64)&v46, a2, a3, a4, a5);
    LOWORD(a5) = v46;
    v48.m128i_i16[0] = v13;
    v8 = v13;
    v48.m128i_i64[1] = v14;
    if ( !(_WORD)v46 )
    {
      v9 = sub_3007100((__int64)&v46);
      v53 = sub_3007260((__int64)&v46);
      v54 = v15;
      v42 = v53;
      if ( v8 )
        goto LABEL_6;
LABEL_11:
      v51 = sub_3007260((__int64)&v48);
      v52 = v16;
      v11 = v51;
      v12 = v52;
      goto LABEL_12;
    }
  }
  if ( (_WORD)a5 == 1 || (unsigned __int16)(a5 - 504) <= 7u )
LABEL_64:
    BUG();
  v9 = (unsigned __int16)(a5 - 176) <= 0x34u;
  v42 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a5 - 16];
  if ( !v8 )
    goto LABEL_11;
LABEL_6:
  if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
    goto LABEL_64;
  v10 = 16LL * (v8 - 1);
  v11 = *(_QWORD *)&byte_444C4A0[v10];
  v12 = byte_444C4A0[v10 + 8];
LABEL_12:
  v57 = v11;
  LOBYTE(v58) = v12;
  v17 = sub_CA1930(&v57);
  v18 = _mm_loadu_si128(&v48);
  v43 = v17;
  v49 = v18;
  v40 = 8 * a7;
  if ( !v9 )
  {
    if ( v41 == v17 )
    {
      a1[1].m128i_i8[0] = 1;
      *a1 = v18;
      return a1;
    }
    v31 = 9;
    v32 = &byte_444C4A0[112];
    v33 = 128;
    while ( 1 )
    {
      v34 = v32[24];
      v50.m128i_i16[0] = v31;
      v50.m128i_i64[1] = 0;
      v57 = v33;
      LOBYTE(v58) = v34;
      v35 = sub_CA1930(&v57);
      if ( v50.m128i_i16[0] )
      {
        if ( v50.m128i_i16[0] == 1 || (unsigned __int16)(v50.m128i_i16[0] - 504) <= 7u )
          goto LABEL_64;
        v37 = 16LL * (v50.m128i_u16[0] - 1);
        v36 = *(_QWORD *)&byte_444C4A0[v37];
        LOBYTE(v37) = byte_444C4A0[v37 + 8];
      }
      else
      {
        v36 = sub_3007260((__int64)&v50);
        v55 = v36;
        v56 = v37;
      }
      v57 = v36;
      LOBYTE(v58) = v37;
      if ( sub_CA1930(&v57) <= (unsigned __int64)v43 )
      {
LABEL_59:
        v9 = 0;
        goto LABEL_13;
      }
      sub_2FE6CC0((__int64)&v57, a3, *(_QWORD *)(a2 + 64), v50.m128i_i64[0], v50.m128i_i64[1]);
      if ( (unsigned __int8)v57 <= 1u )
      {
        v38 = v35;
        if ( !(v42 % v35)
          && v42 >= v35
          && ((v42 / v35) & (v42 / v35 - 1)) == 0
          && (v41 >= v35 || a7 && v40 >= v35 && a8 + v41 >= v35) )
        {
          break;
        }
      }
      if ( --v31 == 1 )
        goto LABEL_59;
      v33 = *(_QWORD *)v32;
      v32 -= 16;
    }
    v9 = 0;
    if ( v42 == v38 )
    {
      v39 = _mm_loadu_si128(&v50);
      a1[1].m128i_i8[0] = 1;
      *a1 = v39;
      return a1;
    }
    v49 = _mm_loadu_si128(&v50);
  }
LABEL_13:
  v19 = (__int64 *)&byte_444C4A0[3632];
  for ( i = 228; i != 16; --i )
  {
    while ( 1 )
    {
      v50.m128i_i16[0] = i;
      v50.m128i_i64[1] = 0;
      if ( (unsigned __int16)(i - 176) <= 0x34u == v9 )
      {
        v44 = *v19;
        sub_2FE6CC0((__int64)&v57, a3, *(_QWORD *)(a2 + 64), v50.m128i_i64[0], 0);
        if ( (unsigned __int8)v57 <= 1u )
        {
          if ( v50.m128i_i16[0] )
          {
            v24 = word_4456580[v50.m128i_u16[0] - 1];
            v25 = 0;
          }
          else
          {
            v24 = sub_3009970((__int64)&v50, a3, v21, v22, v23);
          }
          if ( v48.m128i_i16[0] == v24 )
            break;
        }
      }
LABEL_14:
      --i;
      v19 -= 2;
      if ( i == 16 )
        goto LABEL_35;
    }
    if ( v24 || v25 == v48.m128i_i64[1] )
    {
      if ( !(v42 % (unsigned int)v44)
        && v42 >= (unsigned int)v44
        && ((v42 / (unsigned int)v44) & (v42 / (unsigned int)v44 - 1)) == 0
        && (v41 >= (unsigned int)v44 || a7 && v40 >= (unsigned int)v44 && a8 + v41 >= (unsigned int)v44) )
      {
        v26 = sub_2D5B750((unsigned __int16 *)&v49);
        v58 = v27;
        v57 = v26;
        if ( (unsigned int)v44 > v26 || v50.m128i_i16[0] == (_WORD)v46 && (v50.m128i_i16[0] || v50.m128i_i64[1] == v47) )
        {
          v28 = _mm_loadu_si128(&v50);
          a1[1].m128i_i8[0] = 1;
          *a1 = v28;
          return a1;
        }
      }
      goto LABEL_14;
    }
    v19 -= 2;
  }
LABEL_35:
  if ( v9 )
  {
    a1[1].m128i_i8[0] = 0;
  }
  else
  {
    v29 = _mm_loadu_si128(&v49);
    a1[1].m128i_i8[0] = 1;
    *a1 = v29;
  }
  return a1;
}
