// Function: sub_ED40D0
// Address: 0xed40d0
//
unsigned __int64 *__fastcall sub_ED40D0(unsigned __int64 *a1, __int64 a2, __int64 a3, void *a4, size_t a5, char a6)
{
  __int64 v9; // rax
  __m128i *v10; // rsi
  void *v12; // rax
  size_t v13; // rdx
  void *v14; // r9
  size_t v15; // r8
  int v16; // eax
  __int64 v17; // r9
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rsi
  void *v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  size_t v25; // [rsp+10h] [rbp-50h]
  size_t v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  __int64 v30[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_ED3D70(v30, a2, a4, a5);
  if ( (v30[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v30[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v30[0] = a3;
    v9 = sub_B2F650((__int64)a4, a5);
    v10 = *(__m128i **)(a2 + 104);
    v29 = v9;
    if ( v10 == *(__m128i **)(a2 + 112) )
    {
      sub_ED30D0((const __m128i **)(a2 + 96), v10, &v29, v30);
    }
    else
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = v9;
        v10->m128i_i64[1] = v30[0];
        v10 = *(__m128i **)(a2 + 104);
      }
      *(_QWORD *)(a2 + 104) = v10 + 1;
    }
    if ( a6 )
    {
      v12 = (void *)sub_ED18E0((__int64)a4, a5);
      v14 = v12;
      v15 = v13;
      if ( v13 != a5 || (v25 = v13, a5) && (v21 = v12, v16 = memcmp(v12, a4, a5), v14 = v21, v15 = v25, v16) )
      {
        v22 = (__int64)v14;
        v26 = v15;
        sub_ED3D70(v30, a2, v14, v15);
        v17 = v22;
        v18 = v30[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v30[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v30[0] = 0;
          *a1 = v18 | 1;
          sub_9C66B0(v30);
          return a1;
        }
        v23 = v26;
        v27 = v17;
        v30[0] = 0;
        sub_9C66B0(v30);
        v30[0] = a3;
        v19 = sub_B2F650(v27, v23);
        v20 = *(__m128i **)(a2 + 104);
        v29 = v19;
        if ( v20 == *(__m128i **)(a2 + 112) )
        {
          sub_ED30D0((const __m128i **)(a2 + 96), v20, &v29, v30);
        }
        else
        {
          if ( v20 )
          {
            v20->m128i_i64[0] = v19;
            v20->m128i_i64[1] = v30[0];
            v20 = *(__m128i **)(a2 + 104);
          }
          *(_QWORD *)(a2 + 104) = v20 + 1;
        }
      }
      *a1 = 1;
      v30[0] = 0;
      sub_9C66B0(v30);
    }
    else
    {
      *a1 = 1;
    }
  }
  return a1;
}
