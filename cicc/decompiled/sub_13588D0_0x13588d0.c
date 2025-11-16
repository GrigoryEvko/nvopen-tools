// Function: sub_13588D0
// Address: 0x13588d0
//
bool __fastcall sub_13588D0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4, _QWORD *a5)
{
  char v5; // al
  __int64 v6; // rbx
  __int64 v9; // rdi
  __int64 v10; // r10
  __m128i v11; // xmm0
  __int64 v12; // rax
  __m128i v13; // rax
  __int64 v14; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  __int64 v34; // [rsp+28h] [rbp-88h]
  __m128i v35; // [rsp+30h] [rbp-80h]
  __int64 v36; // [rsp+40h] [rbp-70h]
  __m128i v37; // [rsp+50h] [rbp-60h] BYREF
  __m128i v38; // [rsp+60h] [rbp-50h]
  __int64 v39; // [rsp+70h] [rbp-40h]
  char v40; // [rsp+78h] [rbp-38h]

  v5 = *(_BYTE *)(a1 + 67);
  if ( (v5 & 8) != 0 )
    return 1;
  v6 = *(_QWORD *)(a1 + 16);
  if ( (v5 & 0x40) != 0 )
  {
    for ( ; v6; v6 = *(_QWORD *)(v6 + 16) )
    {
      v13 = *(__m128i *)(v6 + 40);
      v14 = *(_QWORD *)(v6 + 56);
      if ( (v13.m128i_i64[0] == -8 || v13.m128i_i64[0] == -16) && !v13.m128i_i64[1] && !v14 )
        v13.m128i_i64[0] = 0;
      v9 = *(_QWORD *)(v6 + 32);
      v10 = *(_QWORD *)v6;
      v38 = v13;
      v11 = _mm_loadu_si128(a4);
      v37.m128i_i64[1] = v9;
      v33 = a2;
      v39 = v14;
      v34 = a3;
      v12 = a4[1].m128i_i64[0];
      v37.m128i_i64[0] = v10;
      v36 = v12;
      v35 = v11;
      if ( (unsigned __int8)sub_134CB50((__int64)a5, (__int64)&v33, (__int64)&v37) )
        return 1;
    }
    v21 = *(_QWORD *)(a1 + 48);
    v22 = *(_QWORD *)(a1 + 40);
    if ( v21 != v22 )
    {
      v23 = 0xAAAAAAAAAAAAAAABLL * ((v21 - v22) >> 3);
      if ( (_DWORD)v23 )
      {
        v24 = 24LL * (unsigned int)v23;
        v25 = 0;
        while ( 1 )
        {
          v26 = *(_QWORD *)(v22 + v25 + 16);
          if ( v26 )
          {
            v27 = a4->m128i_i64[0];
            v40 = 1;
            v28 = a4->m128i_i64[1];
            v29 = a4[1].m128i_i64[0];
            v37.m128i_i64[0] = a2;
            v38.m128i_i64[0] = v27;
            v38.m128i_i64[1] = v28;
            v37.m128i_i64[1] = a3;
            v39 = v29;
            if ( (sub_13575E0(a5, v26, &v37, a3) & 3) != 0 )
              break;
          }
          v25 += 24;
          if ( v25 == v24 )
            return 0;
          v22 = *(_QWORD *)(a1 + 40);
        }
        return 1;
      }
    }
    return 0;
  }
  else
  {
    v16 = a4[1].m128i_i64[0];
    v37.m128i_i64[0] = a2;
    v37.m128i_i64[1] = a3;
    v39 = v16;
    v38 = _mm_loadu_si128(a4);
    v17 = *(_QWORD *)(v6 + 40);
    v18 = *(_QWORD *)(v6 + 56);
    if ( (v17 == -8 || v17 == -16) && !*(_QWORD *)(v6 + 48) && !v18 )
      v17 = *(_QWORD *)(v6 + 48);
    v19 = *(_QWORD *)(v6 + 32);
    v20 = *(_QWORD *)v6;
    v35.m128i_i64[1] = *(_QWORD *)(v6 + 48);
    v36 = v18;
    v34 = v19;
    v33 = v20;
    v35.m128i_i64[0] = v17;
    return (unsigned __int8)sub_134CB50((__int64)a5, (__int64)&v33, (__int64)&v37) != 0;
  }
}
