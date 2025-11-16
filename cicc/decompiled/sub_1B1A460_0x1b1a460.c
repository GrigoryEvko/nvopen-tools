// Function: sub_1B1A460
// Address: 0x1b1a460
//
_QWORD *__fastcall sub_1B1A460(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        __int64 *a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8,
        __int64 a9,
        __int64 *a10,
        __int64 a11)
{
  __int64 v12; // rax
  unsigned int v13; // r15d
  int v14; // r15d
  __int64 *v15; // rax
  __m128i v16; // xmm3
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rax
  __int64 *v18; // rax
  __m128i v19; // xmm5
  void (__fastcall *v20)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v21; // rdx
  _QWORD *v22; // r12
  __int64 *v24; // rax
  void (__fastcall *v25)(_QWORD, _QWORD, _QWORD); // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  __m128i v28; // xmm4
  __int64 *v29; // rax
  __m128i v30; // xmm6
  void (__fastcall *v31)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v32; // [rsp+0h] [rbp-80h] BYREF
  __int64 *v33; // [rsp+8h] [rbp-78h] BYREF
  int v34; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v35; // [rsp+18h] [rbp-68h] BYREF
  __m128i v36; // [rsp+20h] [rbp-60h] BYREF
  void *v37; // [rsp+30h] [rbp-50h]
  __int64 (__fastcall *v38)(_QWORD); // [rsp+38h] [rbp-48h]
  __m128i v39; // [rsp+40h] [rbp-40h] BYREF
  void (__fastcall *v40)(_QWORD, _QWORD, _QWORD); // [rsp+50h] [rbp-30h]
  __int64 (__fastcall *v41)(_QWORD); // [rsp+58h] [rbp-28h]

  v32 = a5;
  v33 = a4;
  v12 = sub_1599EF0(**(__int64 ****)(*a4 + 16));
  v37 = 0;
  v35 = v12;
  v34 = -1;
  switch ( a3 )
  {
    case 0xBu:
      v36.m128i_i64[0] = a1;
      v14 = 0;
      v36.m128i_i64[1] = (__int64)&v33;
      v37 = sub_1B157C0;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15670;
      goto LABEL_14;
    case 0xCu:
      v40 = 0;
      v26 = (__int64 *)sub_22077B0(32);
      if ( v26 )
      {
        *v26 = a1;
        v26[1] = (__int64)&v35;
        v26[2] = (__int64)&v33;
        v26[3] = (__int64)&v34;
      }
      v39.m128i_i64[0] = (__int64)v26;
      a7 = _mm_loadu_si128(&v36);
      a6 = _mm_loadu_si128(&v39);
      v25 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
      v37 = sub_1B15AE0;
      v39 = a7;
      v40 = v25;
      v41 = v38;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B156C0;
      v36 = a6;
      if ( v25 )
        goto LABEL_23;
      goto LABEL_24;
    case 0xDu:
    case 0xEu:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x22u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
    case 0x27u:
    case 0x28u:
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x2Fu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
    case 0x34u:
      v40 = 0;
      if ( (_BYTE)v32 )
      {
        v18 = (__int64 *)sub_22077B0(24);
        if ( v18 )
        {
          *v18 = a1;
          v18[1] = (__int64)&v33;
          v18[2] = (__int64)&v32;
        }
        v39.m128i_i64[0] = (__int64)v18;
        v19 = _mm_loadu_si128(&v36);
        a6 = _mm_loadu_si128(&v39);
        v20 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
        v37 = sub_1B15920;
        v39 = v19;
        v40 = v20;
        v41 = v38;
        v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15780;
        v36 = a6;
        if ( v20 )
          v20(&v39, &v39, 3);
        v14 = 6;
      }
      else
      {
        v29 = (__int64 *)sub_22077B0(24);
        if ( v29 )
        {
          *v29 = a1;
          v29[1] = (__int64)&v33;
          v29[2] = (__int64)&v32;
        }
        v39.m128i_i64[0] = (__int64)v29;
        v30 = _mm_loadu_si128(&v36);
        a6 = _mm_loadu_si128(&v39);
        v31 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
        v37 = sub_1B158B0;
        v39 = v30;
        v40 = v31;
        v41 = v38;
        v38 = (__int64 (__fastcall *)(_QWORD))sub_1B157A0;
        v36 = a6;
        if ( v31 )
          v31(&v39, &v39, 3);
        v14 = 5;
      }
      goto LABEL_14;
    case 0xFu:
      v36.m128i_i64[0] = a1;
      v14 = 0;
      v36.m128i_i64[1] = (__int64)&v33;
      v37 = sub_1B157F0;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15680;
      goto LABEL_14;
    case 0x10u:
      v40 = 0;
      v24 = (__int64 *)sub_22077B0(32);
      if ( v24 )
      {
        *v24 = a1;
        v24[1] = (__int64)&v35;
        v24[2] = (__int64)&v33;
        v24[3] = (__int64)&v34;
      }
      v39.m128i_i64[0] = (__int64)v24;
      a8 = _mm_loadu_si128(&v36);
      a6 = _mm_loadu_si128(&v39);
      v25 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
      v37 = sub_1B15A70;
      v39 = a8;
      v40 = v25;
      v41 = v38;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15700;
      v36 = a6;
      if ( v25 )
LABEL_23:
        v25(&v39, &v39, 3);
LABEL_24:
      v14 = 0;
      goto LABEL_14;
    case 0x1Au:
      v36.m128i_i64[0] = a1;
      v14 = 0;
      v36.m128i_i64[1] = (__int64)&v33;
      v37 = sub_1B15820;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15690;
      goto LABEL_14;
    case 0x1Bu:
      v36.m128i_i64[0] = a1;
      v14 = 0;
      v36.m128i_i64[1] = (__int64)&v33;
      v37 = sub_1B15850;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B156A0;
      goto LABEL_14;
    case 0x1Cu:
      v36.m128i_i64[0] = a1;
      v14 = 0;
      v36.m128i_i64[1] = (__int64)&v33;
      v37 = sub_1B15880;
      v38 = (__int64 (__fastcall *)(_QWORD))sub_1B156B0;
      goto LABEL_14;
    case 0x33u:
      v13 = BYTE1(v32) == 0 ? 0xFFFFFFFE : 0;
      if ( (_BYTE)v32 )
      {
        v14 = v13 + 4;
        v40 = 0;
        v15 = (__int64 *)sub_22077B0(24);
        if ( v15 )
        {
          *v15 = a1;
          v15[1] = (__int64)&v33;
          v15[2] = (__int64)&v32;
        }
        v39.m128i_i64[0] = (__int64)v15;
        v16 = _mm_loadu_si128(&v36);
        a6 = _mm_loadu_si128(&v39);
        v17 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
        v37 = sub_1B15A00;
        v39 = v16;
        v40 = v17;
        v41 = v38;
        v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15740;
        v36 = a6;
        if ( v17 )
LABEL_6:
          v17(&v39, &v39, 3);
      }
      else
      {
        v14 = v13 + 3;
        v40 = 0;
        v27 = (__int64 *)sub_22077B0(24);
        if ( v27 )
        {
          *v27 = a1;
          v27[1] = (__int64)&v33;
          v27[2] = (__int64)&v32;
        }
        v39.m128i_i64[0] = (__int64)v27;
        v28 = _mm_loadu_si128(&v36);
        a6 = _mm_loadu_si128(&v39);
        v17 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v37;
        v37 = sub_1B15990;
        v39 = v28;
        v40 = v17;
        v41 = v38;
        v38 = (__int64 (__fastcall *)(_QWORD))sub_1B15760;
        v36 = a6;
        if ( v17 )
          goto LABEL_6;
      }
LABEL_14:
      if ( (unsigned __int8)sub_14A3A50(a2, a3, *v33, (unsigned __int16)v32 | (BYTE2(v32) << 16)) )
      {
        if ( !v37 )
          sub_4263D6(a2, a3, v21);
        v22 = (_QWORD *)v38(&v36);
      }
      else
      {
        v22 = sub_1B19ED0(
                a1,
                (__int64)v33,
                a3,
                v14,
                a10,
                a11,
                *(double *)a6.m128i_i64,
                *(double *)a7.m128i_i64,
                *(double *)a8.m128i_i64);
      }
      if ( v37 )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v37)(&v36, &v36, 3);
      return v22;
  }
}
