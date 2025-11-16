// Function: sub_18F2E70
// Address: 0x18f2e70
//
__m128i *__fastcall sub_18F2E70(__m128i *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  char v8; // cl
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rcx
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // [rsp+0h] [rbp-40h] BYREF
  __m128i v17; // [rsp+10h] [rbp-30h] BYREF
  __int64 v18; // [rsp+20h] [rbp-20h]

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 <= 0x17u )
    goto LABEL_2;
  if ( v2 != 78 )
  {
    if ( v2 == 29 )
    {
      v4 = a2 & 0xFFFFFFFFFFFFFFFBLL;
      goto LABEL_7;
    }
LABEL_2:
    a1->m128i_i64[0] = 0;
    a1->m128i_i64[1] = -1;
    a1[1].m128i_i64[0] = 0;
    a1[1].m128i_i64[1] = 0;
    a1[2].m128i_i64[0] = 0;
    return a1;
  }
  v7 = *(_QWORD *)(a2 - 24);
  if ( !*(_BYTE *)(v7 + 16) )
  {
    v8 = *(_BYTE *)(v7 + 33);
    if ( (v8 & 0x20) != 0 )
    {
      if ( (unsigned int)(*(_DWORD *)(v7 + 36) - 133) <= 5 )
      {
        sub_141F750(&v16, a2);
        v14 = _mm_loadu_si128(&v16);
        v15 = _mm_loadu_si128(&v17);
        a1[2].m128i_i64[0] = v18;
        *a1 = v14;
        a1[1] = v15;
        return a1;
      }
      if ( (v8 & 0x20) != 0 )
      {
        v9 = *(_DWORD *)(v7 + 36);
        if ( v9 == 109 )
        {
          v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          goto LABEL_21;
        }
        if ( v9 == 116 )
        {
          v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          v11 = *(_QWORD *)(a2 - 24 * v10);
          v12 = *(_QWORD **)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
            v12 = (_QWORD *)*v12;
          a1->m128i_i64[1] = (__int64)v12;
          v13 = *(_QWORD *)(a2 + 24 * (1 - v10));
          a1[1].m128i_i64[0] = 0;
          a1[1].m128i_i64[1] = 0;
          a1->m128i_i64[0] = v13;
          a1[2].m128i_i64[0] = 0;
          return a1;
        }
        goto LABEL_2;
      }
    }
  }
  v4 = a2 | 4;
LABEL_7:
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v5 )
    goto LABEL_2;
  v6 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
LABEL_21:
  a1->m128i_i64[0] = v6;
  a1->m128i_i64[1] = -1;
  a1[1].m128i_i64[0] = 0;
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i64[0] = 0;
  return a1;
}
