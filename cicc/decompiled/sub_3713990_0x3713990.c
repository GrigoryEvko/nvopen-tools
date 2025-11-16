// Function: sub_3713990
// Address: 0x3713990
//
unsigned __int64 *__fastcall sub_3713990(unsigned __int64 *a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v4; // r13
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __m128i *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rbx
  __m128i *v17; // r8
  __int64 v19; // rdi
  unsigned __int64 *v21; // [rsp+8h] [rbp-88h]
  unsigned __int64 v22; // [rsp+8h] [rbp-88h]
  __m128i *v23; // [rsp+8h] [rbp-88h]
  char v24; // [rsp+1Fh] [rbp-71h] BYREF
  unsigned __int64 v25; // [rsp+20h] [rbp-70h] BYREF
  __int64 v26; // [rsp+28h] [rbp-68h] BYREF
  __m128i v27; // [rsp+30h] [rbp-60h] BYREF
  __m128i v28; // [rsp+40h] [rbp-50h] BYREF
  __int16 v29; // [rsp+50h] [rbp-40h]

  v4 = a2 + 2;
  v7 = a2[9];
  v24 = 1;
  v21 = a4 + 1;
  if ( !v7 )
  {
    v14 = a2[7];
    goto LABEL_22;
  }
  v8 = a2[7];
  if ( v8 )
    goto LABEL_3;
  if ( !a2[8] )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7) )
    {
      v19 = a2[9];
      v27.m128i_i64[0] = (__int64)"Method";
      v29 = 259;
      (*(void (__fastcall **)(__int64, __m128i *))(*(_QWORD *)v19 + 24LL))(v19, &v27);
    }
    v14 = a2[7];
    if ( a2[9] )
    {
      if ( !v14 )
        goto LABEL_18;
      v8 = a2[7];
LABEL_3:
      *(__int64 *)((char *)v27.m128i_i64 + 2) = 0;
      v27.m128i_i16[5] = 0;
      v28 = 0u;
      if ( !*(_BYTE *)(v8 + 48) )
        goto LABEL_15;
LABEL_4:
      v9 = *(_QWORD *)(v8 + 40);
      while ( *(_QWORD *)(v8 + 56) != v9 && (unsigned __int8)sub_1254BC0(a2[7]) <= 0xEFu )
      {
        sub_3713670(&v26, &v24, v4, (__int64)&v27);
        v10 = v26 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v22 = v26 & 0xFFFFFFFFFFFFFFFELL;
          v26 = 0;
          v25 = v10 | 1;
          sub_9C66B0(&v26);
          v11 = v22;
          goto LABEL_9;
        }
        v12 = (__m128i *)a4[2];
        if ( v12 == (__m128i *)a4[3] )
        {
          sub_3711B70(v21, v12, &v27);
        }
        else
        {
          if ( v12 )
          {
            *v12 = _mm_loadu_si128(&v27);
            v12[1] = _mm_loadu_si128(&v28);
            v12 = (__m128i *)a4[2];
          }
          a4[2] = v12 + 2;
        }
        v8 = a2[7];
        if ( *(_BYTE *)(v8 + 48) )
          goto LABEL_4;
LABEL_15:
        v13 = *(_QWORD *)(v8 + 24);
        v9 = 0;
        if ( v13 )
          v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 40LL))(v13) - *(_QWORD *)(v8 + 32);
      }
LABEL_28:
      *a1 = 1;
      return a1;
    }
LABEL_22:
    v8 = v14;
    if ( a2[8] && !v14 )
      goto LABEL_24;
    goto LABEL_3;
  }
LABEL_18:
  if ( a2[8] )
    goto LABEL_3;
LABEL_24:
  v15 = a4[1];
  v16 = a4[2];
  if ( v15 == v16 )
    goto LABEL_28;
  v17 = &v27;
  while ( 1 )
  {
    v23 = v17;
    sub_3713670(v17->m128i_i64, &v24, v4, v15);
    v17 = v23;
    if ( (v27.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      break;
    v15 += 32;
    if ( v16 == v15 )
      goto LABEL_28;
  }
  v25 = 0;
  v27.m128i_i64[0] = v27.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  sub_9C6670((__int64 *)&v25, v23);
  sub_9C66B0(v23->m128i_i64);
  v11 = v25 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v25 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_28;
LABEL_9:
  *a1 = v11 | 1;
  return a1;
}
