// Function: sub_1708300
// Address: 0x1708300
//
_QWORD *__fastcall sub_1708300(const __m128i *a1, unsigned __int8 *a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int8 *v6; // rbx
  unsigned __int8 *v7; // r15
  int v8; // eax
  int v9; // r13d
  unsigned int v10; // eax
  unsigned int v11; // r11d
  unsigned __int8 *v12; // rax
  unsigned int v13; // r11d
  _QWORD *result; // rax
  unsigned __int8 *v15; // r8
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rsi
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  __int64 ***v22; // rax
  __m128i v23; // xmm7
  __int64 ***v24; // rax
  __int64 ***v25; // r11
  unsigned int v26; // r10d
  __int64 v27; // rcx
  int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // rdx
  unsigned __int8 *v31; // rsi
  unsigned __int8 *v32; // rax
  __int64 ***v33; // rax
  __m128i v34; // xmm3
  __int64 ***v35; // rax
  unsigned int v36; // esi
  __int64 ***v37; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v38; // [rsp+10h] [rbp-A0h]
  unsigned int v39; // [rsp+10h] [rbp-A0h]
  unsigned int v40; // [rsp+18h] [rbp-98h]
  int v41; // [rsp+18h] [rbp-98h]
  __int64 ***v42; // [rsp+18h] [rbp-98h]
  unsigned int v43; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v44; // [rsp+20h] [rbp-90h]
  unsigned int v45; // [rsp+20h] [rbp-90h]
  unsigned int v46; // [rsp+28h] [rbp-88h]
  __int64 v47; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v48; // [rsp+28h] [rbp-88h]
  __int64 v49; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v50; // [rsp+38h] [rbp-78h] BYREF
  unsigned __int8 *v51; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int8 *v52; // [rsp+48h] [rbp-68h] BYREF
  __m128i v53; // [rsp+50h] [rbp-60h] BYREF
  __m128i v54; // [rsp+60h] [rbp-50h]
  unsigned __int8 *v55; // [rsp+70h] [rbp-40h]

  v6 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
  v7 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
  v8 = v7[16];
  v9 = a2[16] - 24;
  if ( (unsigned __int8)(v6[16] - 35) > 0x11u )
  {
    if ( (unsigned __int8)v8 <= 0x17u || (unsigned int)(v8 - 35) > 0x11 )
      return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
    v47 = *((_QWORD *)a2 - 3);
    v40 = sub_1704330(
            v9,
            v47,
            &v52,
            v53.m128i_i64,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64);
    if ( v6[16] <= 0x10u )
      goto LABEL_25;
    v44 = 0;
  }
  else
  {
    if ( (unsigned __int8)v8 <= 0x17u || (unsigned int)(v8 - 35) > 0x11 )
    {
      v16 = sub_1704330(
              v9,
              *((_QWORD *)a2 - 6),
              &v50,
              (__int64 *)&v51,
              *(double *)a3.m128i_i64,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64);
      if ( v7[16] <= 0x10u
        || (v45 = v16, v12 = (unsigned __int8 *)sub_15A14F0(v16, *(__int64 ***)v7, 0), v13 = v45, v47 = 0, !v12) )
      {
LABEL_19:
        v44 = v6;
        v47 = 0;
LABEL_20:
        v41 = v44[16] - 24;
        if ( sub_17044F0(v41, v9) )
        {
          v18 = *(unsigned __int8 **)(v17 - 24);
          v19 = *(unsigned __int8 **)(v17 - 48);
          v20 = _mm_loadu_si128(a1 + 167);
          v55 = a2;
          v21 = _mm_loadu_si128(a1 + 168);
          v38 = v18;
          v53 = v20;
          v54 = v21;
          v22 = (__int64 ***)sub_13E1140(v9, v19, v7, &v53);
          v55 = a2;
          v37 = v22;
          v23 = _mm_loadu_si128(a1 + 168);
          v53 = _mm_loadu_si128(a1 + 167);
          v54 = v23;
          v24 = (__int64 ***)sub_13E1140(v9, v38, v7, &v53);
          v25 = v37;
          v26 = v41;
          v27 = (__int64)v24;
          if ( v37 )
          {
            if ( v24 )
              goto LABEL_49;
            if ( v37 == (__int64 ***)sub_15A14F0(v41, *v37, 0) )
            {
              v29 = a1->m128i_i64[1];
              v27 = (__int64)v7;
              v54.m128i_i16[0] = 257;
              v30 = (__int64)v38;
              goto LABEL_38;
            }
          }
          else if ( v24 && v24 == (__int64 ***)sub_15A14F0(v41, *v24, 0) )
          {
            v29 = a1->m128i_i64[1];
            v27 = (__int64)v7;
            v54.m128i_i16[0] = 257;
            v30 = (__int64)v19;
LABEL_38:
            v36 = v9;
LABEL_50:
            v49 = sub_17066B0(
                    v29,
                    v36,
                    v30,
                    v27,
                    v53.m128i_i64,
                    0,
                    *(double *)a3.m128i_i64,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64);
            sub_164B7C0(v49, (__int64)a2);
            return (_QWORD *)v49;
          }
        }
        if ( !v47 )
          return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
LABEL_25:
        v28 = *(unsigned __int8 *)(v47 + 16);
        if ( v9 == 26 )
        {
          if ( (unsigned int)(v28 - 51) > 1 )
            return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
        }
        else if ( v9 == 27 )
        {
          if ( v28 != 50 )
            return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
        }
        else if ( v9 != 15 || ((*(_BYTE *)(v47 + 16) - 35) & 0xFD) != 0 )
        {
          return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
        }
        v39 = v28 - 24;
        a3 = _mm_loadu_si128(a1 + 167);
        a4 = _mm_loadu_si128(a1 + 168);
        v31 = *(unsigned __int8 **)(v47 - 48);
        v32 = *(unsigned __int8 **)(v47 - 24);
        v55 = a2;
        v53 = a3;
        v48 = v32;
        v54 = a4;
        v33 = (__int64 ***)sub_13E1140(v9, v6, v31, &v53);
        v55 = a2;
        a5 = _mm_loadu_si128(a1 + 167);
        v42 = v33;
        v34 = _mm_loadu_si128(a1 + 168);
        v53 = a5;
        v54 = v34;
        v35 = (__int64 ***)sub_13E1140(v9, v6, v48, &v53);
        v25 = v42;
        v26 = v39;
        v27 = (__int64)v35;
        if ( !v42 )
        {
          if ( v35 && v35 == (__int64 ***)sub_15A14F0(v39, *v35, 0) )
          {
            v29 = a1->m128i_i64[1];
            v54.m128i_i16[0] = 257;
            v27 = (__int64)v31;
            goto LABEL_37;
          }
          return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
        }
        if ( !v35 )
        {
          if ( v42 == (__int64 ***)sub_15A14F0(v39, *v42, 0) )
          {
            v29 = a1->m128i_i64[1];
            v54.m128i_i16[0] = 257;
            v27 = (__int64)v48;
LABEL_37:
            v30 = (__int64)v6;
            goto LABEL_38;
          }
          return sub_1707FD0(a1, a2, (__int64)v6, (__int64)v7);
        }
LABEL_49:
        v29 = a1->m128i_i64[1];
        v30 = (__int64)v25;
        v54.m128i_i16[0] = 257;
        v36 = v26;
        goto LABEL_50;
      }
LABEL_8:
      result = (_QWORD *)sub_17068B0(a1, (__int64)a2, v13, v50, v51, v7, a3, a4, *(double *)a5.m128i_i64, v12);
      if ( result )
        return result;
      if ( v47 )
      {
LABEL_40:
        v44 = v6;
        if ( v6[16] <= 0x10u )
          goto LABEL_20;
        goto LABEL_14;
      }
      goto LABEL_19;
    }
    v46 = sub_1704330(
            v9,
            *((_QWORD *)a2 - 6),
            &v50,
            (__int64 *)&v51,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64);
    v10 = sub_1704330(
            v9,
            (__int64)v7,
            &v52,
            v53.m128i_i64,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64);
    v11 = v46;
    v40 = v10;
    if ( v10 == v46 )
    {
      result = (_QWORD *)sub_17068B0(
                           a1,
                           (__int64)a2,
                           v10,
                           v50,
                           v51,
                           v52,
                           a3,
                           a4,
                           *(double *)a5.m128i_i64,
                           (unsigned __int8 *)v53.m128i_i64[0]);
      v11 = v46;
      if ( result )
        return result;
    }
    if ( v7[16] > 0x10u )
    {
      v43 = v11;
      v12 = (unsigned __int8 *)sub_15A14F0(v11, *(__int64 ***)v7, 0);
      v47 = (__int64)v7;
      if ( !v12 )
        goto LABEL_40;
      v13 = v43;
      goto LABEL_8;
    }
    if ( v6[16] <= 0x10u )
    {
      v47 = (__int64)v7;
      v44 = v6;
      goto LABEL_20;
    }
    v44 = v6;
    v47 = (__int64)v7;
  }
LABEL_14:
  v15 = (unsigned __int8 *)sub_15A14F0(v40, *(__int64 ***)v6, 0);
  if ( !v15
    || (result = (_QWORD *)sub_17068B0(
                             a1,
                             (__int64)a2,
                             v40,
                             v6,
                             v15,
                             v52,
                             a3,
                             a4,
                             *(double *)a5.m128i_i64,
                             (unsigned __int8 *)v53.m128i_i64[0])) == 0 )
  {
    if ( !v44 )
      goto LABEL_25;
    goto LABEL_20;
  }
  return result;
}
