// Function: sub_3297000
// Address: 0x3297000
//
__int64 __fastcall sub_3297000(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // r15
  __m128i v8; // xmm1
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int16 v12; // r14
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // r12
  __m128i v18; // xmm3
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdi
  int v24; // eax
  int v25; // r9d
  __int64 v26; // rax
  int v27; // r9d
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __m128i v29; // [rsp+10h] [rbp-A0h] BYREF
  __m128i v30; // [rsp+20h] [rbp-90h] BYREF
  int v31; // [rsp+34h] [rbp-7Ch]
  __int64 v32; // [rsp+38h] [rbp-78h]
  unsigned int v33; // [rsp+40h] [rbp-70h] BYREF
  __int64 v34; // [rsp+48h] [rbp-68h]
  __int64 v35; // [rsp+50h] [rbp-60h] BYREF
  int v36; // [rsp+58h] [rbp-58h]
  _OWORD v37[5]; // [rsp+60h] [rbp-50h] BYREF

  v31 = *(_DWORD *)(a2 + 24);
  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128((const __m128i *)v4);
  v7 = *v4;
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v9 = v4[5];
  v10 = *((unsigned int *)v4 + 2);
  v30 = v6;
  v32 = v9;
  v28 = v10;
  v11 = *(_QWORD *)(v7 + 48) + 16 * v10;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v35 = v5;
  v29 = v8;
  LOWORD(v33) = v12;
  v34 = v13;
  if ( v5 )
    sub_B96E90((__int64)&v35, v5, 1);
  v14 = *(_DWORD *)(v7 + 24) == 51;
  v15 = *a1;
  v36 = *(_DWORD *)(a2 + 72);
  if ( v14 || *(_DWORD *)(v32 + 24) == 51 )
  {
    v16 = sub_34015B0(v15, &v35, v33, v34, 0, 0);
    goto LABEL_6;
  }
  v18 = _mm_load_si128(&v29);
  v37[0] = _mm_load_si128(&v30);
  v37[1] = v18;
  v19 = sub_3402EA0(v15, v31, (unsigned int)&v35, v33, v34, 0, (__int64)v37, 2);
  if ( v19 )
    goto LABEL_10;
  if ( !(unsigned __int8)sub_33E2390(*a1, v30.m128i_i64[0], v30.m128i_i64[1], 1)
    || (unsigned __int8)sub_33E2390(*a1, v29.m128i_i64[0], v29.m128i_i64[1], 1) )
  {
    if ( v12 )
    {
      if ( (unsigned __int16)(v12 - 17) > 0xD3u )
        goto LABEL_14;
    }
    else if ( !sub_30070B0((__int64)&v33) )
    {
      goto LABEL_14;
    }
    v19 = sub_3295970(a1, a2, (__int64)&v35, v20, v21);
    if ( v19 )
    {
LABEL_10:
      v16 = v19;
      goto LABEL_6;
    }
    if ( (unsigned __int8)sub_33D1AE0(v32, 0) )
    {
LABEL_23:
      v19 = v30.m128i_i64[0];
      goto LABEL_10;
    }
LABEL_14:
    if ( !(unsigned __int8)sub_33CF170(v29.m128i_i64[0], v29.m128i_i64[1]) )
    {
      v23 = *a1;
      if ( v31 == 82 )
        v24 = sub_33DF4A0(v23, v7, v28, v32, v29.m128i_u32[2]);
      else
        v24 = sub_33DD440(v23, v7, v28, v32, v29.m128i_u32[2], v22);
      v16 = 0;
      if ( v24 )
        goto LABEL_6;
      v26 = sub_3406EB0(*a1, 56, (unsigned int)&v35, v33, v34, v25, *(_OWORD *)&v30, *(_OWORD *)&v29);
      goto LABEL_19;
    }
    goto LABEL_23;
  }
  v26 = sub_3406EB0(*a1, v31, (unsigned int)&v35, v33, v34, v27, *(_OWORD *)&v29, *(_OWORD *)&v30);
LABEL_19:
  v16 = v26;
LABEL_6:
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v16;
}
