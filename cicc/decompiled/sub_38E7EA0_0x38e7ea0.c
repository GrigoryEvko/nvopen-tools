// Function: sub_38E7EA0
// Address: 0x38e7ea0
//
__m128i *__fastcall sub_38E7EA0(__m128i *a1, char *a2, __int64 a3)
{
  __m128i *v3; // r12
  char *v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  char *v8; // r10
  char v9; // r10
  unsigned __int64 v10; // rcx
  size_t v11; // rax
  unsigned __int64 v12; // rdx
  size_t v13; // r11
  char v14; // dl
  char v15; // al
  unsigned __int8 v16; // si
  char v17; // cl
  char v18; // dl
  char v19; // dl
  char v20; // bl
  unsigned __int8 v21; // cl
  char v22; // dl
  unsigned __int64 v23; // rcx
  size_t v24; // r13
  unsigned __int64 v25; // rdx
  size_t v26; // r11
  __m128i *v27; // rax
  size_t v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  char v30; // [rsp+10h] [rbp-60h]
  size_t v32; // [rsp+18h] [rbp-58h]
  size_t v33; // [rsp+18h] [rbp-58h]
  __m128i *v34; // [rsp+20h] [rbp-50h] BYREF
  __int64 v35; // [rsp+28h] [rbp-48h]
  __m128i v36[4]; // [rsp+30h] [rbp-40h] BYREF

  v3 = a1 + 1;
  if ( a3 )
  {
    v5 = a2;
    v6 = a3;
    v34 = v36;
    v35 = 0;
    v36[0].m128i_i8[0] = 0;
    sub_2240E30((__int64)&v34, (unsigned __int64)(a3 + 1) >> 1);
    v7 = a3;
    v8 = a2;
    if ( (a3 & 1) == 0 )
      goto LABEL_39;
    v19 = *a2;
    v20 = *a2 - 48;
    if ( (unsigned __int8)v20 > 9u )
    {
      v20 = v19 - 87;
      if ( (unsigned __int8)(v19 - 97) > 5u )
      {
        v21 = v19 - 65;
        v20 = -1;
        v22 = v19 - 55;
        if ( v21 < 6u )
          v20 = v22;
      }
    }
    v23 = (unsigned __int64)v34;
    v24 = v35;
    v25 = 15;
    if ( v34 != v36 )
      v25 = v36[0].m128i_i64[0];
    v26 = v35 + 1;
    if ( v35 + 1 > v25 )
    {
      v29 = a3;
      v33 = v35 + 1;
      sub_2240BB0((unsigned __int64 *)&v34, v35, 0, 0, 1u);
      v23 = (unsigned __int64)v34;
      v26 = v33;
      v8 = a2;
      v7 = v29;
    }
    *(_BYTE *)(v23 + v24) = v20;
    v35 = v26;
    v6 = v7 - 1;
    v34->m128i_i8[v24 + 1] = 0;
    v5 = v8 + 1;
    if ( v7 != 1 )
    {
LABEL_39:
      do
      {
        v14 = *v5;
        v15 = v5[1];
        if ( (unsigned __int8)(*v5 - 48) <= 9u )
        {
          v17 = 16 * (v14 - 48);
        }
        else if ( (unsigned __int8)(v14 - 97) <= 5u )
        {
          v17 = 16 * v14 - 112;
        }
        else
        {
          v16 = v14 - 65;
          v17 = -16;
          v18 = 16 * v14 - 112;
          if ( v16 < 6u )
            v17 = v18;
        }
        if ( (unsigned __int8)(v15 - 48) > 9u )
        {
          if ( (unsigned __int8)(v15 - 97) <= 5u )
          {
            v9 = v17 | (v15 - 87);
          }
          else
          {
            v9 = -1;
            if ( (unsigned __int8)(v15 - 65) < 6u )
              v9 = v17 | (v15 - 55);
          }
        }
        else
        {
          v9 = (v15 - 48) | v17;
        }
        v10 = (unsigned __int64)v34;
        v11 = v35;
        v12 = 15;
        if ( v34 != v36 )
          v12 = v36[0].m128i_i64[0];
        v13 = v35 + 1;
        if ( v35 + 1 > v12 )
        {
          v28 = v35 + 1;
          v30 = v9;
          v32 = v35;
          sub_2240BB0((unsigned __int64 *)&v34, v35, 0, 0, 1u);
          v10 = (unsigned __int64)v34;
          v13 = v28;
          v9 = v30;
          v11 = v32;
        }
        *(_BYTE *)(v10 + v11) = v9;
        v35 = v13;
        v34->m128i_i8[v11 + 1] = 0;
        if ( v6 == 1 )
          break;
        v5 += 2;
        v6 -= 2;
      }
      while ( v6 );
    }
    v27 = v34;
    a1->m128i_i64[0] = (__int64)v3;
    if ( v27 == v36 )
    {
      a1[1] = _mm_load_si128(v36);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v27;
      a1[1].m128i_i64[0] = v36[0].m128i_i64[0];
    }
    a1->m128i_i64[1] = v35;
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v3;
    a1->m128i_i64[1] = 0;
    a1[1].m128i_i8[0] = 0;
  }
  return a1;
}
