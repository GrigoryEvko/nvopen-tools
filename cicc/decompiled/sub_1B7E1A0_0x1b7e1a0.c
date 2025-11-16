// Function: sub_1B7E1A0
// Address: 0x1b7e1a0
//
__int64 __fastcall sub_1B7E1A0(char *a1, char *a2, __int64 a3, const __m128i *a4)
{
  __int64 result; // rax
  char *v5; // rbx
  char *v6; // r14
  __int64 v7; // rdx
  char *v8; // r12
  __int64 v9; // r13
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  char *v15; // r15
  _QWORD *v16; // r14
  unsigned __int8 (__fastcall *v17)(_QWORD *, __int64, __int64, __int64); // rax
  char *v18; // rax
  _QWORD *v19; // rbx
  char *v20; // r14
  unsigned __int8 (__fastcall *v21)(_QWORD *, __int64, _QWORD, _QWORD); // rax
  char *v22; // rax
  __int64 v23; // rax
  __m128i v24; // xmm2
  unsigned __int8 v25; // al
  char *v26; // r13
  char *v27; // r14
  char *v28; // r15
  char *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // r12
  __int64 v32; // rcx
  __int64 *v33; // r12
  __int64 v34; // rcx
  unsigned __int8 v35; // al
  bool v36; // zf
  char *v37; // [rsp+10h] [rbp-D0h]
  __int64 v39; // [rsp+20h] [rbp-C0h]
  char *v40; // [rsp+28h] [rbp-B8h]
  __int64 v41; // [rsp+28h] [rbp-B8h]
  __int64 *v42; // [rsp+30h] [rbp-B0h]
  char *v43; // [rsp+38h] [rbp-A8h]
  __int64 v44; // [rsp+38h] [rbp-A8h]
  __int64 v45; // [rsp+40h] [rbp-A0h]
  _OWORD v46[2]; // [rsp+50h] [rbp-90h] BYREF
  char *v47; // [rsp+70h] [rbp-70h] BYREF
  char *v48; // [rsp+78h] [rbp-68h]
  char *v49; // [rsp+80h] [rbp-60h]
  char *v50; // [rsp+88h] [rbp-58h]
  char *v51[10]; // [rsp+90h] [rbp-50h] BYREF

  result = a2 - a1;
  v40 = a2;
  v39 = a3;
  if ( a2 - a1 <= 128 )
    return result;
  v5 = a1;
  if ( !a3 )
  {
    v42 = (__int64 *)a2;
    goto LABEL_30;
  }
  v37 = a1 + 8;
  while ( 2 )
  {
    --v39;
    v6 = &v5[8 * (result >> 4)];
    v7 = *((_QWORD *)v5 + 1);
    v8 = (char *)a4->m128i_i64[0];
    v9 = a4[1].m128i_i64[0];
    v10 = (_QWORD *)(a4[1].m128i_i64[1] + a4->m128i_i64[1]);
    v11 = *(_QWORD *)v6;
    v45 = a4->m128i_i64[0] & 1;
    if ( (a4->m128i_i64[0] & 1) == 0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64, __int64))v8)(v10, v9, v7, v11) )
      {
        if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v8)(
               v10,
               v9,
               *(_QWORD *)v6,
               *((_QWORD *)v40 - 1)) )
        {
          goto LABEL_7;
        }
        v25 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v8)(
                v10,
                v9,
                *((_QWORD *)v5 + 1),
                *((_QWORD *)v40 - 1));
LABEL_27:
        if ( v25 )
        {
          v12 = *(_QWORD *)v5;
          goto LABEL_40;
        }
LABEL_36:
        v13 = *(_QWORD *)v5;
        v14 = *((_QWORD *)v5 + 1);
        *((_QWORD *)v5 + 1) = *(_QWORD *)v5;
        *(_QWORD *)v5 = v14;
        goto LABEL_9;
      }
      if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v8)(
             v10,
             v9,
             *((_QWORD *)v5 + 1),
             *((_QWORD *)v40 - 1)) )
      {
        goto LABEL_36;
      }
      v35 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v8)(
              v10,
              v9,
              *(_QWORD *)v6,
              *((_QWORD *)v40 - 1));
LABEL_39:
      v36 = v35 == 0;
      v12 = *(_QWORD *)v5;
      if ( v36 )
        goto LABEL_8;
LABEL_40:
      *(_QWORD *)v5 = *((_QWORD *)v40 - 1);
      *((_QWORD *)v40 - 1) = v12;
      v13 = *((_QWORD *)v5 + 1);
      v14 = *(_QWORD *)v5;
      goto LABEL_9;
    }
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, __int64, __int64))&v8[*v10 - 1])(v10, v9, v7, v11) )
    {
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))&v8[*v10 - 1])(
             v10,
             v9,
             *((_QWORD *)v5 + 1),
             *((_QWORD *)v40 - 1)) )
      {
        goto LABEL_36;
      }
      v35 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))&v8[*v10 - 1])(
              v10,
              v9,
              *(_QWORD *)v6,
              *((_QWORD *)v40 - 1));
      goto LABEL_39;
    }
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))&v8[*v10 - 1])(
            v10,
            v9,
            *(_QWORD *)v6,
            *((_QWORD *)v40 - 1)) )
    {
      v25 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))&v8[*v10 - 1])(
              v10,
              v9,
              *((_QWORD *)v5 + 1),
              *((_QWORD *)v40 - 1));
      goto LABEL_27;
    }
LABEL_7:
    v12 = *(_QWORD *)v5;
LABEL_8:
    *(_QWORD *)v5 = *(_QWORD *)v6;
    *(_QWORD *)v6 = v12;
    v13 = *((_QWORD *)v5 + 1);
    v14 = *(_QWORD *)v5;
LABEL_9:
    v43 = v37;
    v15 = v40;
    v16 = v10;
    while ( 1 )
    {
      v42 = (__int64 *)v43;
      v17 = (unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64, __int64))v8;
      if ( v45 )
        v17 = *(unsigned __int8 (__fastcall **)(_QWORD *, __int64, __int64, __int64))&v8[*v16 - 1];
      if ( !v17(v16, v9, v13, v14) )
        break;
LABEL_20:
      v14 = *(_QWORD *)v5;
      v13 = *((_QWORD *)v43 + 1);
      v43 += 8;
    }
    v18 = v5;
    v15 -= 8;
    v19 = v16;
    v20 = v18;
    while ( 1 )
    {
      v21 = (unsigned __int8 (__fastcall *)(_QWORD *, __int64, _QWORD, _QWORD))v8;
      if ( v45 )
        v21 = *(unsigned __int8 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))&v8[*v19 - 1];
      if ( !v21(v19, v9, *(_QWORD *)v20, *(_QWORD *)v15) )
        break;
      v15 -= 8;
    }
    v22 = v20;
    v16 = v19;
    v5 = v22;
    if ( v43 < v15 )
    {
      v23 = *(_QWORD *)v43;
      *(_QWORD *)v43 = *(_QWORD *)v15;
      *(_QWORD *)v15 = v23;
      goto LABEL_20;
    }
    v24 = _mm_loadu_si128(a4 + 1);
    v46[0] = _mm_loadu_si128(a4);
    v46[1] = v24;
    sub_1B7E1A0(v43, v40, v39, v46);
    result = v43 - v5;
    if ( v43 - v5 > 128 )
    {
      if ( v39 )
      {
        v40 = v43;
        continue;
      }
LABEL_30:
      v41 = result >> 3;
      v27 = (char *)a4->m128i_i64[0];
      v28 = (char *)a4->m128i_i64[1];
      v29 = (char *)a4[1].m128i_i64[1];
      v44 = ((result >> 3) - 2) >> 1;
      v30 = *(_QWORD *)&v5[8 * v44];
      v49 = (char *)a4[1].m128i_i64[0];
      v26 = v49;
      v50 = v29;
      v47 = v27;
      v48 = v28;
      sub_1B7D9A0((__int64)v5, v44, result >> 3, v30, &v47);
      v31 = v44;
      do
      {
        --v31;
        v47 = v27;
        v32 = *(_QWORD *)&v5[8 * v31];
        v48 = v28;
        v49 = v26;
        v50 = v29;
        sub_1B7D9A0((__int64)v5, v31, v41, v32, &v47);
      }
      while ( v31 );
      v33 = v42;
      do
      {
        v34 = *--v33;
        *v33 = *(_QWORD *)v5;
        v51[0] = v27;
        v51[1] = v28;
        v51[2] = v26;
        v51[3] = v29;
        result = sub_1B7D9A0((__int64)v5, 0, ((char *)v33 - v5) >> 3, v34, v51);
      }
      while ( (char *)v33 - v5 > 8 );
    }
    return result;
  }
}
