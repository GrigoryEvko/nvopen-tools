// Function: sub_117B180
// Address: 0x117b180
//
unsigned __int8 *__fastcall sub_117B180(__int64 **a1, char a2, __int64 a3, _BYTE *a4)
{
  const __m128i *v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  unsigned __int8 *v10; // r12
  unsigned __int8 *v11; // rbx
  __int64 v12; // r14
  unsigned __int8 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rcx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __m128i v31[2]; // [rsp+10h] [rbp-80h] BYREF
  __m128i v32; // [rsp+30h] [rbp-60h]
  __m128i v33; // [rsp+40h] [rbp-50h]
  __int64 v34; // [rsp+50h] [rbp-40h]

  v6 = (const __m128i *)a1[2];
  v7 = a1[3];
  v31[0] = _mm_loadu_si128(v6 + 6);
  v31[1] = _mm_loadu_si128(v6 + 7);
  v32 = _mm_loadu_si128(v6 + 8);
  v33 = _mm_loadu_si128(v6 + 9);
  v8 = v6[10].m128i_i64[0];
  v32.m128i_i64[1] = (__int64)v7;
  v9 = a1[1];
  v34 = v8;
  v10 = (unsigned __int8 *)sub_1020DD0((__int64)a4, **a1, *v9, v31);
  if ( v10 )
  {
    v32.m128i_i16[0] = 257;
    if ( a2 )
    {
      v12 = (__int64)v10;
      v11 = (unsigned __int8 *)*a1[1];
    }
    else
    {
      v11 = v10;
      v12 = **a1;
    }
    v13 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v10 = v13;
    if ( v13 )
    {
      sub_B44260((__int64)v13, *(_QWORD *)(v12 + 8), 57, 3u, 0, 0);
      if ( *((_QWORD *)v10 - 12) )
      {
        v14 = *((_QWORD *)v10 - 11);
        **((_QWORD **)v10 - 10) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *((_QWORD *)v10 - 10);
      }
      *((_QWORD *)v10 - 12) = a3;
      if ( a3 )
      {
        v15 = *(_QWORD *)(a3 + 16);
        *((_QWORD *)v10 - 11) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = v10 - 88;
        *((_QWORD *)v10 - 10) = a3 + 16;
        *(_QWORD *)(a3 + 16) = v10 - 96;
      }
      if ( *((_QWORD *)v10 - 8) )
      {
        v16 = *((_QWORD *)v10 - 7);
        **((_QWORD **)v10 - 6) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *((_QWORD *)v10 - 6);
      }
      *((_QWORD *)v10 - 8) = v12;
      v17 = *(_QWORD *)(v12 + 16);
      *((_QWORD *)v10 - 7) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = v10 - 56;
      *((_QWORD *)v10 - 6) = v12 + 16;
      *(_QWORD *)(v12 + 16) = v10 - 64;
      if ( *((_QWORD *)v10 - 4) )
      {
        v18 = *((_QWORD *)v10 - 3);
        **((_QWORD **)v10 - 2) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *((_QWORD *)v10 - 2);
      }
      *((_QWORD *)v10 - 4) = v11;
      if ( v11 )
      {
        v19 = *((_QWORD *)v11 + 2);
        *((_QWORD *)v10 - 3) = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = v10 - 24;
        *((_QWORD *)v10 - 2) = v11 + 16;
        *((_QWORD *)v11 + 2) = v10 - 32;
      }
      sub_BD6B50(v10, (const char **)v31);
    }
  }
  else
  {
    v21 = *(_QWORD *)(*a1[4] + 16);
    if ( v21 && !*(_QWORD *)(v21 + 8) )
    {
      v22 = *a1[5];
      v23 = *(unsigned __int8 *)(v22 + 8);
      if ( (unsigned int)(v23 - 17) <= 1 )
        LOBYTE(v23) = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
      if ( (_BYTE)v23 == 12 && *a4 == 82 )
      {
        v24 = **a1;
        v25 = *(_QWORD *)(v24 + 8);
        v26 = *(unsigned __int8 *)(v25 + 8);
        if ( (unsigned int)(v26 - 17) <= 1 )
          LOBYTE(v26) = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
        if ( (_BYTE)v26 == 12 )
        {
          v27 = sub_1179770((__int64)a4, v24, (unsigned __int8 *)*a1[1], (__int64)a1[2]);
          if ( v27 )
          {
            v32.m128i_i16[0] = 257;
            if ( a2 )
            {
              v28 = v27;
              v29 = *a1[1];
            }
            else
            {
              v28 = **a1;
              v29 = v27;
            }
            return sub_109FEA0(a3, v28, v29, (const char **)v31, 0, 0, 0);
          }
        }
      }
    }
  }
  return v10;
}
