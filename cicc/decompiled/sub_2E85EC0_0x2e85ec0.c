// Function: sub_2E85EC0
// Address: 0x2e85ec0
//
__m128i *__fastcall sub_2E85EC0(__m128i *a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __m128i *v6; // r12
  void (__fastcall *v7)(_BYTE *, __int64, __int64); // rax
  __int64 v8; // rax
  void (__fastcall *v9)(_QWORD *, _BYTE *, __int64); // rax
  void (__fastcall *v10)(_QWORD *, _QWORD *, __int64); // rax
  void (__fastcall *v11)(_BYTE *, __int64, __int64); // rax
  __int64 v12; // rax
  void (__fastcall *v13)(_QWORD *, _BYTE *, __int64); // rax
  void (__fastcall *v14)(_QWORD *, _QWORD *, __int64); // rax
  _BYTE *v15; // rax
  __m128i v16; // xmm0
  __m128i v17; // xmm2
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __m128i v22; // xmm1
  __int64 v23; // rdx
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v26; // [rsp+8h] [rbp-138h]
  _BYTE v27[16]; // [rsp+10h] [rbp-130h] BYREF
  __m128i v28; // [rsp+20h] [rbp-120h]
  _BYTE v29[16]; // [rsp+30h] [rbp-110h] BYREF
  __m128i v30; // [rsp+40h] [rbp-100h]
  _BYTE *v31; // [rsp+50h] [rbp-F0h]
  _BYTE *v32; // [rsp+58h] [rbp-E8h]
  __m128i v33; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v34; // [rsp+70h] [rbp-D0h]
  _BYTE *v35; // [rsp+80h] [rbp-C0h]
  _BYTE *v36; // [rsp+88h] [rbp-B8h]
  __m128i v37; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v38; // [rsp+A0h] [rbp-A0h]
  _QWORD v39[2]; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v40; // [rsp+C0h] [rbp-80h]
  _QWORD v41[2]; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v42[5]; // [rsp+F0h] [rbp-50h] BYREF

  v6 = a1;
  v7 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 16);
  v26 = a2;
  v30.m128i_i64[0] = 0;
  if ( !v7 )
    goto LABEL_31;
  a2 = (_BYTE *)a4;
  a1 = (__m128i *)v29;
  v7(v29, a4, 2);
  v8 = *(_QWORD *)(a4 + 24);
  v42[0].m128i_i64[0] = 0;
  v30.m128i_i64[1] = v8;
  v9 = *(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(a4 + 16);
  v30.m128i_i64[0] = (__int64)v9;
  if ( v9 )
  {
    a2 = v29;
    a1 = (__m128i *)v41;
    v9(v41, v29, 2);
    v35 = a3;
    v36 = a3;
    v38.m128i_i64[0] = 0;
    v42[0] = v30;
    if ( v30.m128i_i64[0] )
    {
      a1 = &v37;
      ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v30.m128i_i64[0])(&v37, v41, 2);
      a2 = v35;
      v10 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v42[0].m128i_i64[0];
      v38 = v42[0];
      if ( v36 != v35 )
      {
        while ( 1 )
        {
          if ( !v10 )
            goto LABEL_35;
          a1 = &v37;
          if ( ((unsigned __int8 (__fastcall *)(__m128i *))v38.m128i_i64[1])(&v37) )
            break;
          a2 = v35 + 40;
          v35 = a2;
          if ( v36 == a2 )
            break;
          v10 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v38.m128i_i64[0];
        }
        v10 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v42[0].m128i_i64[0];
      }
      if ( v10 )
      {
        a2 = v41;
        a1 = (__m128i *)v41;
        v10(v41, v41, 3);
      }
    }
  }
  else
  {
LABEL_31:
    v35 = a3;
    v36 = a3;
    v38.m128i_i64[0] = 0;
  }
  v28.m128i_i64[0] = 0;
  v11 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 16);
  if ( !v11 )
  {
    v40.m128i_i64[0] = 0;
    goto LABEL_33;
  }
  a2 = (_BYTE *)a4;
  a1 = (__m128i *)v27;
  v11(v27, a4, 2);
  v12 = *(_QWORD *)(a4 + 24);
  v40.m128i_i64[0] = 0;
  v28.m128i_i64[1] = v12;
  v13 = *(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(a4 + 16);
  v28.m128i_i64[0] = (__int64)v13;
  if ( !v13 )
  {
LABEL_33:
    v32 = a3;
    v34.m128i_i64[0] = 0;
    v31 = v26;
    goto LABEL_34;
  }
  a2 = v27;
  a1 = (__m128i *)v39;
  v13(v39, v27, 2);
  v32 = a3;
  v34.m128i_i64[0] = 0;
  v31 = v26;
  v40 = v28;
  if ( v28.m128i_i64[0] )
  {
    a1 = &v33;
    ((void (__fastcall *)(__m128i *, _QWORD *, __int64))v28.m128i_i64[0])(&v33, v39, 2);
    a2 = v31;
    v14 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v40.m128i_i64[0];
    v34 = v40;
    if ( v31 != v32 )
    {
      while ( 1 )
      {
        if ( !v14 )
          goto LABEL_35;
        a1 = &v33;
        if ( ((unsigned __int8 (__fastcall *)(__m128i *))v34.m128i_i64[1])(&v33) )
          break;
        a2 = v31 + 40;
        v31 = a2;
        if ( v32 == a2 )
          break;
        v14 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v34.m128i_i64[0];
      }
      v14 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v40.m128i_i64[0];
    }
    if ( v14 )
      v14(v39, v39, 3);
    goto LABEL_25;
  }
LABEL_34:
  if ( v26 != a3 )
LABEL_35:
    sub_4263D6(a1, a2, a3);
LABEL_25:
  v15 = v35;
  v16 = _mm_loadu_si128(&v37);
  v17 = _mm_loadu_si128(v42);
  v39[0] = v31;
  v18 = v38.m128i_i64[0];
  v6->m128i_i64[0] = (__int64)v31;
  v41[0] = v15;
  v19 = v38.m128i_i64[1];
  v6->m128i_i64[1] = (__int64)v32;
  v20 = v34.m128i_i64[0];
  v6[3].m128i_i64[0] = (__int64)v15;
  v21 = v36;
  v22 = _mm_loadu_si128(&v33);
  v6[2].m128i_i64[0] = v20;
  v6[3].m128i_i64[1] = (__int64)v21;
  v23 = v34.m128i_i64[1];
  v24 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v28.m128i_i64[0];
  v6[5].m128i_i64[0] = v18;
  v38 = 0u;
  v6[2].m128i_i64[1] = v23;
  v6[5].m128i_i64[1] = v19;
  v37 = v17;
  v42[0] = v16;
  v40 = v22;
  v6[1] = v22;
  v6[4] = v16;
  if ( v24 )
  {
    v24(v27, v27, 3);
    if ( v38.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v38.m128i_i64[0])(&v37, &v37, 3);
  }
  if ( v30.m128i_i64[0] )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v30.m128i_i64[0])(v29, v29, 3);
  return v6;
}
