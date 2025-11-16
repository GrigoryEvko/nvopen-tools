// Function: sub_33EE0D0
// Address: 0x33ee0d0
//
_QWORD *__fastcall sub_33EE0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  char v7; // al
  __int64 v8; // rbx
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 *v13; // rsi
  bool v14; // cf
  const __m128i *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // r13
  unsigned int v23; // ebx
  unsigned int v24; // eax
  _QWORD v26[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v27[2]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v28; // [rsp+20h] [rbp-80h] BYREF
  char v29; // [rsp+28h] [rbp-78h]
  unsigned __int64 v30; // [rsp+30h] [rbp-70h] BYREF
  char v31; // [rsp+38h] [rbp-68h]
  __m128i v32; // [rsp+40h] [rbp-60h]
  __int64 v33; // [rsp+50h] [rbp-50h]
  __int64 v34; // [rsp+58h] [rbp-48h]
  __int64 v35; // [rsp+60h] [rbp-40h]
  __int64 v36; // [rsp+68h] [rbp-38h]

  v27[0] = a2;
  v27[1] = a3;
  v26[0] = a4;
  v26[1] = a5;
  if ( (_WORD)a2 )
  {
    if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
      goto LABEL_16;
    v8 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
    v7 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
  }
  else
  {
    v33 = sub_3007260((__int64)v27);
    v7 = v6;
    v8 = v33;
    v34 = v6;
  }
  v29 = v7;
  v9 = (unsigned __int64)(v8 + 7) >> 3;
  v28 = v9;
  if ( !LOWORD(v26[0]) )
  {
    v10 = sub_3007260((__int64)v26);
    v35 = v10;
    v36 = v11;
    goto LABEL_5;
  }
  if ( LOWORD(v26[0]) == 1 || (unsigned __int16)(LOWORD(v26[0]) - 504) <= 7u )
LABEL_16:
    BUG();
  v10 = *(_QWORD *)&byte_444C4A0[16 * LOWORD(v26[0]) - 16];
  LOBYTE(v11) = byte_444C4A0[16 * LOWORD(v26[0]) - 8];
LABEL_5:
  v31 = v11;
  v12 = (unsigned __int64)(v10 + 7) >> 3;
  v13 = *(__int64 **)(a1 + 64);
  v14 = v12 < v9;
  v30 = v12;
  v15 = (const __m128i *)&v30;
  if ( v14 )
    v15 = (const __m128i *)&v28;
  v32 = _mm_loadu_si128(v15);
  v16 = sub_3007410((__int64)v27, v13, (__int64)&v28, (__int64)&v30, a5, a6);
  v21 = sub_3007410((__int64)v26, *(__int64 **)(a1 + 64), v17, v18, v19, v20);
  v22 = sub_2E79000(*(__int64 **)(a1 + 40));
  v23 = sub_AE5260(v22, v21);
  v24 = sub_AE5260(v22, v16);
  if ( (unsigned __int8)v23 > (unsigned __int8)v24 )
    v24 = v23;
  return sub_33EDE90(a1, v32.m128i_i64[0], v32.m128i_i64[1], v24);
}
