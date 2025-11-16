// Function: sub_7F66A0
// Address: 0x7f66a0
//
void __fastcall sub_7F66A0(__m128i **a1)
{
  __m128i *v1; // r15
  __m128i *v2; // rdx
  const __m128i *v4; // r14
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __m128i *v8; // rax
  const __m128i *v9; // rdi
  __int8 v10; // dl
  __int64 v11; // r9
  __m128i *v12; // rax
  __m128i *v13; // rdx
  __m128i *v14; // rax
  __m128i *v15; // r15
  __m128i *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int8 v18; // [rsp+Fh] [rbp-31h]

  v1 = *a1;
  if ( (*a1)[10].m128i_i8[13] == 11 )
  {
    v2 = a1[1];
    v4 = (const __m128i *)v1[11].m128i_i64[0];
    v5 = v1[7].m128i_i64[1];
    v18 = v1[12].m128i_i8[0];
    v6 = (__int64)&v2[-1].m128i_i64[1] + 7;
    v7 = v1[11].m128i_i64[1] - (_QWORD)v2;
    if ( v7 )
    {
      v1[11].m128i_i64[1] = v7;
      if ( v7 == 1 )
        sub_72A510(v4, v1);
      v12 = sub_7401F0((__int64)v4);
      v1[7].m128i_i64[1] = (__int64)v12;
      v11 = (__int64)v12;
    }
    else if ( v18 )
    {
      v8 = sub_7401F0((__int64)v4);
      v9 = v8;
      while ( v8 )
      {
        while ( (v8[10].m128i_i64[1] & 0xFF0000002000LL) == 0xA0000000000LL )
        {
          v13 = (__m128i *)v8[11].m128i_i64[0];
          if ( !v13 )
            break;
          if ( v13[10].m128i_i8[13] == 13 )
            v13 = (__m128i *)v13[7].m128i_i64[1];
          v13[7].m128i_i64[1] = 0;
          v8[11].m128i_i64[1] = (__int64)v13;
          v8 = v13;
        }
        v10 = v8[10].m128i_i8[13];
        if ( v10 == 11 )
        {
          v8 = (__m128i *)v8[11].m128i_i64[0];
        }
        else
        {
          if ( v10 != 13 )
            break;
          v8 = (__m128i *)v8[7].m128i_i64[1];
        }
      }
      sub_72A510(v9, v1);
      v11 = (__int64)v1;
    }
    else
    {
      sub_72A510(v4, v1);
      v11 = (__int64)v1;
    }
    if ( v6 )
    {
      v17 = v11;
      v14 = sub_7401F0((__int64)v4);
      v11 = v17;
      v15 = v14;
      if ( v6 != 1 )
      {
        v16 = (__m128i *)sub_724D50(11);
        v11 = v17;
        v16[11].m128i_i64[1] = v6;
        v16[11].m128i_i64[0] = (__int64)v15;
        if ( (v15[-1].m128i_i8[8] & 8) == 0 )
          v16[-1].m128i_i8[8] &= ~8u;
        v15 = v16;
        v16[12].m128i_i8[0] = v18;
      }
      *(_QWORD *)(v11 + 120) = v15;
      v15[7].m128i_i64[1] = v5;
    }
    else
    {
      *(_QWORD *)(v11 + 120) = v5;
    }
    sub_7F51D0(v11, 1, 0, (__int64)a1);
  }
}
