// Function: sub_6F7980
// Address: 0x6f7980
//
__int64 __fastcall sub_6F7980(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rax
  __m128i v19; // [rsp+0h] [rbp-170h] BYREF
  __m128i v20; // [rsp+10h] [rbp-160h]
  __m128i v21; // [rsp+20h] [rbp-150h]
  __m128i v22; // [rsp+30h] [rbp-140h]
  __m128i v23; // [rsp+40h] [rbp-130h]
  __m128i v24; // [rsp+50h] [rbp-120h]
  __m128i v25; // [rsp+60h] [rbp-110h]
  __m128i v26; // [rsp+70h] [rbp-100h]
  __m128i v27; // [rsp+80h] [rbp-F0h]
  __int64 v28; // [rsp+90h] [rbp-E0h]

  if ( !a1[1].m128i_i8[0] )
    return sub_6E6870((__int64)a1);
  v3 = a1->m128i_i64[0];
  v4 = *(_BYTE *)(v3 + 140);
  if ( v4 == 12 )
  {
    v5 = v3;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v4 = *(_BYTE *)(v5 + 140);
    }
    while ( v4 == 12 );
  }
  if ( !v4 )
    return sub_6E6870((__int64)a1);
  v6 = *(_BYTE *)(a2 + 140);
  if ( v6 == 12 )
  {
    v7 = a2;
    do
    {
      v7 = *(_QWORD *)(v7 + 160);
      v6 = *(_BYTE *)(v7 + 140);
    }
    while ( v6 == 12 );
  }
  if ( !v6 )
    return sub_6E6840((__int64)a1);
  result = sub_8DAAE0(v3, a2);
  if ( !(_DWORD)result )
  {
    result = a1[1].m128i_u8[0];
    if ( (_BYTE)result == 2 )
    {
      a1->m128i_i64[0] = a2;
      a1[17].m128i_i64[0] = a2;
    }
    else
    {
      v19 = _mm_loadu_si128(a1);
      v20 = _mm_loadu_si128(a1 + 1);
      v21 = _mm_loadu_si128(a1 + 2);
      v22 = _mm_loadu_si128(a1 + 3);
      v23 = _mm_loadu_si128(a1 + 4);
      v24 = _mm_loadu_si128(a1 + 5);
      v25 = _mm_loadu_si128(a1 + 6);
      v26 = _mm_loadu_si128(a1 + 7);
      v27 = _mm_loadu_si128(a1 + 8);
      if ( (_BYTE)result == 1 || (_BYTE)result == 5 )
        v28 = a1[9].m128i_i64[0];
      v13 = sub_6F6F40(a1, 0, v9, v10, v11, v12);
      v18 = (__int64 *)sub_73E170(
                         v13,
                         a2,
                         v14,
                         v15,
                         v16,
                         v17,
                         v19.m128i_i64[0],
                         v19.m128i_i64[1],
                         v20.m128i_i64[0],
                         v20.m128i_i64[1],
                         v21.m128i_i64[0],
                         v21.m128i_i64[1],
                         v22.m128i_i64[0],
                         v22.m128i_i64[1],
                         v23.m128i_i64[0],
                         v23.m128i_i64[1],
                         v24.m128i_i64[0],
                         v24.m128i_i64[1],
                         v25.m128i_i64[0],
                         v25.m128i_i64[1],
                         v26.m128i_i64[0],
                         v26.m128i_i64[1],
                         v27.m128i_i64[0],
                         v27.m128i_i64[1],
                         v28);
      sub_6E70E0(v18, (__int64)a1);
      return sub_6E4BC0((__int64)a1, (__int64)&v19);
    }
  }
  return result;
}
