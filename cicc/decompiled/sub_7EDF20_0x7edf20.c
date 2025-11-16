// Function: sub_7EDF20
// Address: 0x7edf20
//
__m128i *__fastcall sub_7EDF20(_QWORD *a1, unsigned int a2, int a3, __int64 a4, __m128i **a5)
{
  __m128i *v8; // r15
  unsigned __int64 v9; // rcx
  __int128 v10; // rax
  __int64 v11; // r8
  int v12; // r14d
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // r15
  __m128i *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rax
  __m128i *result; // rax
  _QWORD *v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rcx
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __m128i *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-E8h]
  __int64 v32; // [rsp+8h] [rbp-E8h]
  __int64 v33; // [rsp+10h] [rbp-E0h]
  __int64 v34; // [rsp+10h] [rbp-E0h]
  __int64 v35; // [rsp+10h] [rbp-E0h]
  __int64 v36; // [rsp+10h] [rbp-E0h]
  __int64 v37; // [rsp+18h] [rbp-D8h]
  __int64 v39; // [rsp+20h] [rbp-D0h]
  __int64 v40; // [rsp+20h] [rbp-D0h]
  __int64 v41; // [rsp+20h] [rbp-D0h]
  __int64 v42; // [rsp+28h] [rbp-C8h]
  __m128i *v43; // [rsp+38h] [rbp-B8h] BYREF
  _BYTE v44[32]; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE v45[144]; // [rsp+60h] [rbp-90h] BYREF

  v8 = (__m128i *)a1[9];
  *(_QWORD *)dword_4D03F38 = *a1;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
  v9 = (unsigned __int64)qword_4D03F68;
  v10 = *(_OWORD *)(a1[10] + 8LL);
  v37 = qword_4D03F68[6];
  v11 = *((_QWORD *)&v10 + 1) | v10;
  if ( v10 == 0 )
  {
    if ( a3 )
    {
      a3 = 1;
      v33 = *((_QWORD *)&v10 + 1) | v10;
      sub_7E18E0((__int64)v45, 0, 0);
      v11 = v33;
      v12 = *((unsigned __int8 *)qword_4D03F68 + 24);
      if ( *((_BYTE *)qword_4D03F68 + 24) )
      {
LABEL_4:
        v31 = v11;
        sub_7E1740((__int64)a1, (__int64)v44);
        sub_7E91D0(v31, (__int64)v44);
      }
    }
    else
    {
      if ( a1 == *(_QWORD **)(qword_4F04C50 + 80LL) )
      {
        v11 = *(_QWORD *)(qword_4F04C50 + 88LL);
        if ( !v11 )
        {
          if ( !a2 )
          {
            sub_7EDD70(v8, &v43);
            goto LABEL_26;
          }
          v12 = 0;
          goto LABEL_40;
        }
        v12 = 1;
        goto LABEL_4;
      }
      v13 = (_QWORD *)qword_4F18A10;
      if ( qword_4F18A10 )
      {
        qword_4F18A10 = *(_QWORD *)qword_4F18A10;
      }
      else
      {
        v13 = (_QWORD *)sub_823970(24);
        v9 = (unsigned __int64)qword_4D03F68;
      }
      v12 = 0;
      *v13 = *(_QWORD *)(v9 + 80);
      *((_QWORD *)&v10 + 1) = qword_4D03F68;
      qword_4D03F68[10] = v13;
      v13[1] = a1;
      v13[2] = *(_QWORD *)(*((_QWORD *)&v10 + 1) + 72LL);
      *(_QWORD *)(*((_QWORD *)&v10 + 1) + 72LL) = 0;
    }
  }
  else
  {
    v32 = *(_QWORD *)(a1[10] + 16LL);
    v34 = *(_QWORD *)(a1[10] + 8LL);
    sub_7E18E0((__int64)v45, v10, *((__int64 *)&v10 + 1));
    *((_QWORD *)&v10 + 1) = v32;
    v12 = *((unsigned __int8 *)qword_4D03F68 + 24);
    if ( v34 )
      v11 = *(_QWORD *)(v34 + 88);
    else
      v11 = v32;
    a3 = 1;
    if ( *((_BYTE *)qword_4D03F68 + 24) )
      goto LABEL_4;
  }
  v14 = *(_QWORD *)(a1[10] + 8LL);
  if ( v14 )
  {
    v15 = *(_QWORD **)(v14 + 32);
    if ( v15 )
      sub_7DDF80(v15);
  }
  v9 = a2;
  if ( a2 )
LABEL_40:
    sub_806920(a1, a4, *((_QWORD *)&v10 + 1), v9, v11);
  sub_7EDD70(v8, &v43);
  if ( !v12 )
    goto LABEL_25;
  v16 = a1[10];
  v17 = v43;
  v18 = *(_QWORD *)(v16 + 8);
  v19 = *(_QWORD *)(v16 + 16);
  if ( !v43 )
  {
    if ( !a1[9]
      || (v40 = *(_QWORD *)(v16 + 16),
          v42 = *(_QWORD *)(v16 + 8),
          v30 = sub_7E2C20((__int64)a1),
          v18 = v42,
          v19 = v40,
          (v17 = (__m128i *)v30) == 0) )
    {
      v36 = v19;
      v41 = v18;
      sub_7E1740((__int64)a1, (__int64)v44);
      v21 = v41;
      v20 = v36;
      v22 = qword_4F04C50;
      if ( a1 == *(_QWORD **)(qword_4F04C50 + 80LL) )
        goto LABEL_49;
LABEL_19:
      if ( !v21 )
        goto LABEL_21;
      goto LABEL_20;
    }
  }
  v35 = v19;
  v39 = v18;
  sub_7E1720((__int64)v17, (__int64)v44);
  v20 = v35;
  v21 = v39;
  v22 = qword_4F04C50;
  if ( a1 != *(_QWORD **)(qword_4F04C50 + 80LL) )
    goto LABEL_19;
LABEL_49:
  v21 = v22;
LABEL_20:
  v20 = *(_QWORD *)(v21 + 88);
LABEL_21:
  if ( (*(_BYTE *)(v16 + 24) & 1) != 0 )
  {
    if ( *(_DWORD *)v16 )
      *(_QWORD *)dword_4D03F38 = *(_QWORD *)v16;
    sub_7E7530(v20, (__int64)v44);
  }
  else if ( v37 != qword_4D03F68[6] )
  {
    qword_4D03F68[6] = v37;
  }
LABEL_25:
  if ( a3 )
  {
    result = (__m128i *)sub_7E1AA0();
    goto LABEL_32;
  }
LABEL_26:
  result = (__m128i *)qword_4F04C50;
  if ( a1 != *(_QWORD **)(qword_4F04C50 + 80LL) )
  {
    v24 = qword_4D03F68;
    v25 = qword_4D03F68[9];
    v26 = (_QWORD *)qword_4D03F68[10];
    if ( v25 )
    {
      v27 = (_QWORD *)qword_4D03F68[9];
      do
      {
        v28 = v27;
        v27 = (_QWORD *)*v27;
      }
      while ( v27 );
      *v28 = qword_4F18A18;
      qword_4F18A18 = v25;
      v24 = qword_4D03F68;
    }
    v24[9] = v26[2];
    v24[10] = *v26;
    result = (__m128i *)qword_4F18A10;
    *v26 = qword_4F18A10;
    qword_4F18A10 = (__int64)v26;
  }
LABEL_32:
  if ( a5 )
  {
    v29 = v43;
    if ( v43 )
    {
      for ( result = (__m128i *)v43[1].m128i_i64[0]; result; result = (__m128i *)result[1].m128i_i64[0] )
        v29 = result;
    }
    *a5 = v29;
  }
  return result;
}
