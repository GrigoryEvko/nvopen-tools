// Function: sub_15804A0
// Address: 0x15804a0
//
__m128i *__fastcall sub_15804A0(__m128i *a1, _BYTE *a2, __int64 a3)
{
  __m128i *v4; // r12
  void (__fastcall *v5)(_BYTE *, __int64, __int64); // rax
  __int64 v6; // rax
  void (__fastcall *v7)(_QWORD *, _BYTE *, __int64); // rax
  _BYTE *v8; // r14
  void (__fastcall *v9)(_QWORD *, _QWORD *, __int64); // rax
  _BYTE *v10; // rdx
  void (__fastcall *v11)(_BYTE *, __int64, __int64); // rax
  __int64 v12; // rax
  void (__fastcall *v13)(_QWORD *, _BYTE *, __int64); // rax
  _BYTE *v14; // r13
  void (__fastcall *v15)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v16; // r9
  _BYTE *v17; // rax
  __int64 v18; // rcx
  __m128i v19; // xmm0
  _BYTE *v20; // rdx
  __m128i v21; // xmm2
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // rsi
  __m128i v25; // xmm1
  __int64 v26; // rcx
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v29; // [rsp+8h] [rbp-138h]
  _BYTE v30[16]; // [rsp+10h] [rbp-130h] BYREF
  __m128i v31; // [rsp+20h] [rbp-120h]
  _BYTE v32[16]; // [rsp+30h] [rbp-110h] BYREF
  __m128i v33; // [rsp+40h] [rbp-100h]
  _BYTE *v34; // [rsp+50h] [rbp-F0h]
  _BYTE *v35; // [rsp+58h] [rbp-E8h]
  __m128i v36; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v37; // [rsp+70h] [rbp-D0h]
  _BYTE *v38; // [rsp+80h] [rbp-C0h]
  _BYTE *v39; // [rsp+88h] [rbp-B8h]
  __m128i v40; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v41; // [rsp+A0h] [rbp-A0h]
  _QWORD v42[2]; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v43; // [rsp+C0h] [rbp-80h]
  _QWORD v44[2]; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v45[5]; // [rsp+F0h] [rbp-50h] BYREF

  v4 = a1;
  v5 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
  v29 = a2;
  v33.m128i_i64[0] = 0;
  if ( v5 )
  {
    a2 = (_BYTE *)a3;
    a1 = (__m128i *)v32;
    v5(v32, a3, 2);
    v6 = *(_QWORD *)(a3 + 24);
    v45[0].m128i_i64[0] = 0;
    v33.m128i_i64[1] = v6;
    v7 = *(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(a3 + 16);
    v8 = v29 + 40;
    v33.m128i_i64[0] = (__int64)v7;
    if ( v7 )
    {
      a2 = v32;
      a1 = (__m128i *)v44;
      v7(v44, v32, 2);
      v38 = v29 + 40;
      v39 = v29 + 40;
      v41.m128i_i64[0] = 0;
      v45[0] = v33;
      if ( v33.m128i_i64[0] )
      {
        a1 = &v40;
        ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v33.m128i_i64[0])(&v40, v44, 2);
        a2 = v38;
        v9 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v45[0].m128i_i64[0];
        v41 = v45[0];
        if ( v38 != v39 )
        {
          while ( 1 )
          {
            v10 = a2 - 24;
            if ( a2 )
              a2 -= 24;
            if ( !v9 )
LABEL_42:
              sub_4263D6(a1, a2, v10);
            a1 = &v40;
            if ( ((unsigned __int8 (__fastcall *)(__m128i *, _BYTE *))v41.m128i_i64[1])(&v40, a2) )
              break;
            a2 = (_BYTE *)*((_QWORD *)v38 + 1);
            v38 = a2;
            if ( v39 == a2 )
              break;
            v9 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v41.m128i_i64[0];
          }
          v9 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v45[0].m128i_i64[0];
        }
        if ( v9 )
        {
          a2 = v44;
          a1 = (__m128i *)v44;
          v9(v44, v44, 3);
        }
      }
    }
    else
    {
      v38 = v29 + 40;
      v39 = v29 + 40;
      v41.m128i_i64[0] = 0;
    }
  }
  else
  {
    v41.m128i_i64[0] = 0;
    v8 = a2 + 40;
    v38 = a2 + 40;
    v39 = a2 + 40;
  }
  v31.m128i_i64[0] = 0;
  v11 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
  if ( !v11 )
  {
    v43.m128i_i64[0] = 0;
    v14 = (_BYTE *)*((_QWORD *)v29 + 6);
    goto LABEL_38;
  }
  a2 = (_BYTE *)a3;
  a1 = (__m128i *)v30;
  v11(v30, a3, 2);
  v12 = *(_QWORD *)(a3 + 24);
  v43.m128i_i64[0] = 0;
  v31.m128i_i64[1] = v12;
  v13 = *(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(a3 + 16);
  v14 = (_BYTE *)*((_QWORD *)v29 + 6);
  v31.m128i_i64[0] = (__int64)v13;
  if ( !v13 )
  {
LABEL_38:
    v34 = v14;
    v35 = v8;
    v37.m128i_i64[0] = 0;
    goto LABEL_39;
  }
  a2 = v30;
  a1 = (__m128i *)v42;
  v13(v42, v30, 2);
  v34 = v14;
  v35 = v8;
  v37.m128i_i64[0] = 0;
  v43 = v31;
  if ( v31.m128i_i64[0] )
  {
    a1 = &v36;
    a2 = v42;
    ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v31.m128i_i64[0])(&v36, v42, 2);
    v14 = v34;
    v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v43.m128i_i64[0];
    v37 = v43;
    if ( v34 != v35 )
      goto LABEL_22;
    goto LABEL_27;
  }
LABEL_39:
  v15 = 0;
  v16 = 0;
  if ( v8 == v14 )
    goto LABEL_30;
  while ( 1 )
  {
LABEL_22:
    v10 = v14 - 24;
    if ( v14 )
      v14 -= 24;
    if ( !v15 )
      goto LABEL_42;
    a2 = v14;
    a1 = &v36;
    if ( ((unsigned __int8 (__fastcall *)(__m128i *, _BYTE *))v37.m128i_i64[1])(&v36, v14) )
      break;
    v14 = (_BYTE *)*((_QWORD *)v34 + 1);
    v34 = v14;
    if ( v35 == v14 )
      break;
    v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v37.m128i_i64[0];
  }
  v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v43.m128i_i64[0];
LABEL_27:
  if ( v15 )
    v15(v42, v42, 3);
  v16 = v37.m128i_i64[0];
LABEL_30:
  v17 = v39;
  v18 = (__int64)v35;
  v4[2].m128i_i64[0] = v16;
  v19 = _mm_loadu_si128(&v40);
  v20 = v38;
  v21 = _mm_loadu_si128(v45);
  v22 = v41.m128i_i64[0];
  v44[1] = v17;
  v23 = v41.m128i_i64[1];
  v24 = (__int64)v34;
  v44[0] = v38;
  v25 = _mm_loadu_si128(&v36);
  v42[1] = v18;
  v4->m128i_i64[1] = v18;
  v26 = v37.m128i_i64[1];
  v4[3].m128i_i64[1] = (__int64)v17;
  v27 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v31.m128i_i64[0];
  v41 = 0u;
  v42[0] = v24;
  v4->m128i_i64[0] = v24;
  v4[2].m128i_i64[1] = v26;
  v4[3].m128i_i64[0] = (__int64)v20;
  v4[5].m128i_i64[0] = v22;
  v4[5].m128i_i64[1] = v23;
  v40 = v21;
  v45[0] = v19;
  v43 = v25;
  v4[1] = v25;
  v4[4] = v19;
  if ( v27 )
  {
    v27(v30, v30, 3);
    if ( v41.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v41.m128i_i64[0])(&v40, &v40, 3);
  }
  if ( v33.m128i_i64[0] )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v33.m128i_i64[0])(v32, v32, 3);
  return v4;
}
