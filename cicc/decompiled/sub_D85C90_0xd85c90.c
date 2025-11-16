// Function: sub_D85C90
// Address: 0xd85c90
//
__m128i *__fastcall sub_D85C90(__int64 a1, __int64 a2)
{
  __m128i *v3; // r12
  unsigned int v4; // eax
  unsigned int v5; // eax
  __int32 v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r14
  __m128i *v9; // rbx
  __m128i *v10; // r13
  unsigned int v11; // eax
  unsigned int v12; // eax
  __int32 v13; // eax
  __int64 v14; // rdi
  unsigned int v16; // eax
  unsigned int v17; // eax

  v3 = (__m128i *)sub_22077B0(80);
  v3[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v4 = *(_DWORD *)(a1 + 56);
  v3[3].m128i_i32[2] = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780((__int64)v3[3].m128i_i64, (const void **)(a1 + 48));
    v17 = *(_DWORD *)(a1 + 72);
    v3[4].m128i_i32[2] = v17;
    if ( v17 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    v3[3].m128i_i64[0] = *(_QWORD *)(a1 + 48);
    v5 = *(_DWORD *)(a1 + 72);
    v3[4].m128i_i32[2] = v5;
    if ( v5 <= 0x40 )
    {
LABEL_3:
      v3[4].m128i_i64[0] = *(_QWORD *)(a1 + 64);
      goto LABEL_4;
    }
  }
  sub_C43780((__int64)v3[4].m128i_i64, (const void **)(a1 + 64));
LABEL_4:
  v6 = *(_DWORD *)a1;
  v7 = *(_QWORD *)(a1 + 24);
  v3[1].m128i_i64[0] = 0;
  v3[1].m128i_i64[1] = 0;
  v3->m128i_i32[0] = v6;
  v3->m128i_i64[1] = a2;
  if ( v7 )
    v3[1].m128i_i64[1] = sub_D85C90(v7, v3);
  v8 = *(_QWORD *)(a1 + 16);
  if ( v8 )
  {
    v9 = v3;
    do
    {
      v10 = v9;
      v9 = (__m128i *)sub_22077B0(80);
      v9[2] = _mm_loadu_si128((const __m128i *)(v8 + 32));
      v11 = *(_DWORD *)(v8 + 56);
      v9[3].m128i_i32[2] = v11;
      if ( v11 > 0x40 )
      {
        sub_C43780((__int64)v9[3].m128i_i64, (const void **)(v8 + 48));
        v16 = *(_DWORD *)(v8 + 72);
        v9[4].m128i_i32[2] = v16;
        if ( v16 > 0x40 )
        {
LABEL_16:
          sub_C43780((__int64)v9[4].m128i_i64, (const void **)(v8 + 64));
          goto LABEL_11;
        }
      }
      else
      {
        v9[3].m128i_i64[0] = *(_QWORD *)(v8 + 48);
        v12 = *(_DWORD *)(v8 + 72);
        v9[4].m128i_i32[2] = v12;
        if ( v12 > 0x40 )
          goto LABEL_16;
      }
      v9[4].m128i_i64[0] = *(_QWORD *)(v8 + 64);
LABEL_11:
      v13 = *(_DWORD *)v8;
      v9[1].m128i_i64[0] = 0;
      v9[1].m128i_i64[1] = 0;
      v9->m128i_i32[0] = v13;
      v10[1].m128i_i64[0] = (__int64)v9;
      v9->m128i_i64[1] = (__int64)v10;
      v14 = *(_QWORD *)(v8 + 24);
      if ( v14 )
        v9[1].m128i_i64[1] = sub_D85C90(v14, v9);
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v8 );
  }
  return v3;
}
