// Function: sub_3785EA0
// Address: 0x3785ea0
//
unsigned __int8 *__fastcall sub_3785EA0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  unsigned __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // r9d
  unsigned __int8 *v12; // rax
  __int64 v13; // rdi
  __int32 v14; // edx
  int v15; // r9d
  unsigned __int8 *v16; // rax
  _QWORD *v17; // rdi
  int v18; // edx
  __int64 v19; // r9
  unsigned __int8 *v20; // r14
  __int32 v22; // edx
  int v23; // edx
  __int64 v24; // r14
  int v25; // r9d
  __int64 v26; // [rsp+50h] [rbp-90h] BYREF
  __int64 v27; // [rsp+58h] [rbp-88h]
  __m128i v28; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int8 *v29; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v30; // [rsp+78h] [rbp-68h]
  __int64 v31; // [rsp+80h] [rbp-60h] BYREF
  int v32; // [rsp+88h] [rbp-58h]
  unsigned int v33; // [rsp+90h] [rbp-50h] BYREF
  __int64 v34; // [rsp+98h] [rbp-48h]
  unsigned int v35; // [rsp+A0h] [rbp-40h]
  __int64 v36; // [rsp+A8h] [rbp-38h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v28.m128i_i64[0] = 0;
  v28.m128i_i32[2] = 0;
  v27 = v7;
  v8 = *(unsigned __int64 **)(a2 + 40);
  LOWORD(v26) = v6;
  LODWORD(v30) = 0;
  v9 = v8[1];
  v29 = 0;
  sub_375E8D0((__int64)a1, *v8, v9, (__int64)&v28, (__int64)&v29);
  v10 = *(_QWORD *)(a2 + 80);
  v31 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v31, v10, 1);
  v32 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v26 )
  {
    if ( (unsigned __int16)(v26 - 176) <= 0x34u )
    {
LABEL_5:
      sub_33D0340((__int64)&v33, a1[1], &v26);
      v12 = sub_33FAF80(a1[1], 234, (__int64)&v31, v33, v34, v11, a3);
      v13 = a1[1];
      v28.m128i_i64[0] = (__int64)v12;
      v28.m128i_i32[2] = v14;
      v16 = sub_33FAF80(v13, 234, (__int64)&v31, v35, v36, v15, a3);
      v17 = (_QWORD *)a1[1];
      v29 = v16;
      LODWORD(v30) = v18;
      v20 = sub_3406EB0(
              v17,
              0x9Fu,
              (__int64)&v31,
              (unsigned int)v26,
              v27,
              v19,
              *(_OWORD *)&v28,
              __PAIR128__(v30, (unsigned __int64)v16));
      goto LABEL_6;
    }
  }
  else if ( sub_3007100((__int64)&v26) )
  {
    goto LABEL_5;
  }
  v28.m128i_i64[0] = (__int64)sub_375A6A0((__int64)a1, v28.m128i_i64[0], v28.m128i_u32[2], a3);
  v28.m128i_i32[2] = v22;
  v29 = sub_375A6A0((__int64)a1, (__int64)v29, v30, a3);
  LODWORD(v30) = v23;
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
  {
    a3 = _mm_loadu_si128(&v28);
    v28.m128i_i64[0] = (__int64)v29;
    v28.m128i_i32[2] = v30;
    v29 = (unsigned __int8 *)a3.m128i_i64[0];
    LODWORD(v30) = a3.m128i_i32[2];
  }
  v24 = a1[1];
  sub_375AFE0(a1, v28.m128i_i64[0], v28.m128i_i64[1], (__int64)v29, v30, a3);
  v20 = sub_33FAF80(v24, 234, (__int64)&v31, (unsigned int)v26, v27, v25, a3);
LABEL_6:
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v20;
}
