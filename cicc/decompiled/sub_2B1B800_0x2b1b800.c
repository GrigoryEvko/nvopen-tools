// Function: sub_2B1B800
// Address: 0x2b1b800
//
__int64 *__fastcall sub_2B1B800(__int64 *a1, __int64 a2, char **a3, __int64 a4)
{
  __int64 *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // r13
  char v17; // al
  __int64 v18; // rdi
  char v19; // al
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  __m128i v24; // [rsp+10h] [rbp-80h] BYREF
  __int64 v25; // [rsp+20h] [rbp-70h]
  __int64 v26; // [rsp+28h] [rbp-68h]
  __int64 v27; // [rsp+30h] [rbp-60h]
  __int64 v28; // [rsp+38h] [rbp-58h]
  __int64 v29; // [rsp+40h] [rbp-50h]
  __int64 v30; // [rsp+48h] [rbp-48h]
  __int16 v31; // [rsp+50h] [rbp-40h]

  v6 = a1;
  v7 = a2 - (_QWORD)a1;
  v8 = v7 >> 3;
  if ( v7 >> 5 > 0 )
  {
    v9 = &a1[4 * (v7 >> 5)];
    do
    {
      if ( **a3 != 13 )
      {
        v14 = *v6;
        v24 = (__m128i)*(unsigned __int64 *)(a4 + 3344);
        v25 = 0;
        v26 = 0;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v30 = 0;
        v31 = 257;
        if ( !(unsigned __int8)sub_9AC470(v14, &v24, 0) )
          return v6;
        v10 = v6[1];
        if ( **a3 != 13 )
        {
          v24 = (__m128i)*(unsigned __int64 *)(a4 + 3344);
          v25 = 0;
          v26 = 0;
          v27 = 0;
          v28 = 0;
          v29 = 0;
          v30 = 0;
          v31 = 257;
          if ( !(unsigned __int8)sub_9AC470(v10, &v24, 0) )
            return v6 + 1;
          v11 = v6[2];
          if ( **a3 != 13 )
          {
            v24 = (__m128i)*(unsigned __int64 *)(a4 + 3344);
            v25 = 0;
            v26 = 0;
            v27 = 0;
            v28 = 0;
            v29 = 0;
            v30 = 0;
            v31 = 257;
            if ( !(unsigned __int8)sub_9AC470(v11, &v24, 0) )
              return v6 + 2;
            v12 = v6[3];
            if ( **a3 != 13 )
            {
              v13 = *(_QWORD *)(a4 + 3344);
              v31 = 257;
              v24 = (__m128i)v13;
              v25 = 0;
              v26 = 0;
              v27 = 0;
              v28 = 0;
              v29 = 0;
              v30 = 0;
              if ( !(unsigned __int8)sub_9AC470(v12, &v24, 0) )
                return v6 + 3;
            }
          }
        }
      }
      v6 += 4;
    }
    while ( v6 != v9 );
    v8 = (a2 - (__int64)v6) >> 3;
  }
  if ( v8 == 2 )
  {
    v15 = v6;
    v19 = **a3;
    goto LABEL_30;
  }
  if ( v8 == 3 )
  {
    v15 = v6 + 1;
    v17 = **a3;
    if ( v17 == 13 )
    {
LABEL_33:
      ++v15;
      goto LABEL_26;
    }
    v18 = *v6;
    v15 = v6;
    v24 = (__m128i)*(unsigned __int64 *)(a4 + 3344);
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 257;
    if ( !(unsigned __int8)sub_9AC470(v18, &v24, 0) )
      return v15;
    v15 = v6 + 1;
    v19 = **a3;
LABEL_30:
    if ( v19 == 13 )
      return (__int64 *)a2;
    v21 = *(_QWORD *)(a4 + 3344);
    v22 = *v15;
    v31 = 257;
    v24 = (__m128i)v21;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    if ( !(unsigned __int8)sub_9AC470(v22, &v24, 0) )
      return v15;
    v17 = **a3;
    goto LABEL_33;
  }
  if ( v8 != 1 )
    return (__int64 *)a2;
  v15 = v6;
  v17 = **a3;
LABEL_26:
  if ( v17 == 13 )
    return (__int64 *)a2;
  v20 = *v15;
  v24 = (__m128i)*(unsigned __int64 *)(a4 + 3344);
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 257;
  if ( (unsigned __int8)sub_9AC470(v20, &v24, 0) )
    return (__int64 *)a2;
  return v15;
}
