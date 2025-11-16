// Function: sub_7E0A10
// Address: 0x7e0a10
//
_QWORD *__fastcall sub_7E0A10(__m128i *a1)
{
  const __m128i *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  const __m128i *v11; // rsi
  __int8 v12; // dl
  int v13; // eax
  const __m128i *v14; // r13
  __int64 v15; // rcx
  const __m128i *v16; // rdi
  __m128i *v17; // rax
  __int64 v18; // rdi
  void *v19; // rax
  const __m128i *v20; // rax
  _QWORD *v22; // rax
  const __m128i *v23; // rax
  void *v24; // rax
  const __m128i *v25; // r14
  const __m128i *v26; // rax
  void *v27; // r15
  __int64 v28; // rax
  const __m128i *v29; // rax
  void *v30; // r13
  __int64 v31; // rax
  const __m128i *v32; // rax
  __int64 v33; // rax
  const __m128i *v34; // [rsp+8h] [rbp-48h]
  const __m128i *v35; // [rsp+18h] [rbp-38h] BYREF

  v2 = (const __m128i *)sub_724DC0();
  v7 = a1[1].m128i_i64[0];
  v35 = v2;
  if ( (a1[1].m128i_i8[9] & 1) == 0 )
    sub_7E0590((__int64)a1);
  if ( !(unsigned int)sub_731770((__int64)a1, 0, v3, v4, v5, v6) )
  {
    v22 = sub_72BA30(5u);
    sub_72BB40((__int64)v22, v35);
    v23 = (const __m128i *)sub_73A720(v35, (__int64)v35);
    sub_730620((__int64)a1, v23);
    goto LABEL_12;
  }
  if ( a1[1].m128i_i8[8] != 1 )
  {
    v19 = sub_730FF0(a1);
    v20 = (const __m128i *)sub_73E1B0((__int64)v19, 0);
    sub_730620((__int64)a1, v20);
    goto LABEL_12;
  }
  v11 = (const __m128i *)a1[4].m128i_i64[1];
  v12 = a1[3].m128i_i8[10];
  v13 = a1[3].m128i_u8[8];
  v14 = (const __m128i *)v11[1].m128i_i64[0];
  if ( (v12 & 1) == 0 )
  {
    if ( (_BYTE)v13 == 94 )
    {
      if ( (v11[1].m128i_i8[9] & 3) != 0 )
      {
        v34 = (const __m128i *)a1[4].m128i_i64[1];
        sub_7E0A10(v34);
        v11 = v34;
      }
    }
    else if ( (_BYTE)v13 != 95 )
    {
      if ( (unsigned __int8)(v13 - 100) <= 1u )
      {
        sub_7E0A10(v11[1].m128i_i64[0]);
        a1[1].m128i_i8[9] &= ~1u;
        v18 = v14->m128i_i64[0];
        a1->m128i_i64[0] = v14->m128i_i64[0];
        goto LABEL_9;
      }
      if ( (unsigned __int8)(v13 - 71) <= 1u )
      {
        v26 = (const __m128i *)sub_731370((__int64)a1, (__int64)v11, (unsigned int)(v13 - 100), v8, v9, v10);
        sub_730620((__int64)a1, v26);
        v18 = a1->m128i_i64[0];
        goto LABEL_9;
      }
      v24 = sub_730FF0(a1);
      v11 = (const __m128i *)sub_73E1B0((__int64)v24, (__int64)v11);
    }
    sub_730620((__int64)a1, v11);
    v18 = a1->m128i_i64[0];
    goto LABEL_9;
  }
  v15 = (unsigned int)(v13 - 103);
  if ( (unsigned __int8)(v13 - 103) <= 1u )
  {
    v25 = (const __m128i *)v14[1].m128i_i64[0];
    if ( v14[1].m128i_i8[8] != 8 )
      sub_7E0A10(v11[1].m128i_i64[0]);
    if ( v25[1].m128i_i8[8] != 8 )
      sub_7E0A10(v25);
    v18 = v25->m128i_i64[0];
    if ( v14->m128i_i64[0] != v25->m128i_i64[0] )
    {
      if ( !(unsigned int)sub_8D97D0(v14->m128i_i64[0], v25->m128i_i64[0], 1, v15, v14->m128i_i64[0]) )
      {
        v27 = sub_730FF0(v14);
        v28 = sub_72CBE0();
        v29 = (const __m128i *)sub_73E110((__int64)v27, v28);
        sub_730620((__int64)v14, v29);
        v30 = sub_730FF0(v25);
        v31 = sub_72CBE0();
        v32 = (const __m128i *)sub_73E110((__int64)v30, v31);
        sub_730620((__int64)v25, v32);
        v33 = sub_72CBE0();
        a1->m128i_i64[0] = v33;
        v18 = v33;
        goto LABEL_28;
      }
      v18 = v14->m128i_i64[0];
    }
    a1->m128i_i64[0] = v18;
LABEL_28:
    a1[1].m128i_i8[9] &= ~1u;
    a1[3].m128i_i8[10] &= ~1u;
    goto LABEL_9;
  }
  if ( (_BYTE)v13 == 91 )
  {
    sub_7E0A10(v11[1].m128i_i64[0]);
    a1[1].m128i_i8[9] &= ~1u;
    a1[3].m128i_i8[10] &= ~1u;
    v18 = v14->m128i_i64[0];
    a1->m128i_i64[0] = v14->m128i_i64[0];
  }
  else
  {
    a1[1].m128i_i8[9] &= ~1u;
    v16 = (const __m128i *)a1->m128i_i64[0];
    a1[3].m128i_i8[10] = v12 & 0xFE;
    v17 = sub_73D720(v16);
    a1->m128i_i64[0] = (__int64)v17;
    v18 = (__int64)v17;
  }
LABEL_9:
  if ( (unsigned int)sub_8D2600(v18) )
    sub_7304E0((__int64)a1);
LABEL_12:
  a1[1].m128i_i64[0] = v7;
  return sub_724E30((__int64)&v35);
}
