// Function: sub_14203A0
// Address: 0x14203a0
//
bool *__fastcall sub_14203A0(bool *a1, __int64 a2, const __m128i *a3, __int64 a4, _QWORD *a5)
{
  unsigned __int8 v6; // al
  __int64 v7; // rsi
  __int64 v8; // r9
  char v9; // di
  unsigned __int16 v10; // cx
  unsigned int v11; // edx
  char v12; // al
  char v14; // al
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // rax
  char v18; // al
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rax
  char v22; // al
  __m128i v23; // [rsp+0h] [rbp-40h] BYREF
  __m128i v24; // [rsp+10h] [rbp-30h]
  __int64 v25; // [rsp+20h] [rbp-20h]
  char v26; // [rsp+28h] [rbp-18h]

  v6 = *(_BYTE *)(a4 + 16);
  v7 = *(_QWORD *)(a2 + 72);
  if ( v6 > 0x17u )
  {
    v8 = a4 | 4;
    if ( v6 == 78 || (v8 = a4 & 0xFFFFFFFFFFFFFFFBLL, v6 == 29) )
    {
      v9 = *(_BYTE *)(v7 + 16);
      if ( v9 != 78 )
      {
        if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_5;
LABEL_12:
        v14 = sub_134F8C0(a5, v7, v8, a4, (__int64)a5);
        a1[2] = 1;
        a1[1] = (v14 & 4) == 0 ? 3 : 1;
        *a1 = (v14 & 3) != 0;
        return a1;
      }
      v19 = *(_QWORD *)(v7 - 24);
      if ( *(_BYTE *)(v19 + 16) || (*(_BYTE *)(v19 + 33) & 0x20) == 0 )
        goto LABEL_14;
LABEL_19:
      v20 = *(_DWORD *)(v19 + 36);
      if ( v20 == 117 )
      {
        if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          v21 = *(_QWORD *)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
          v23.m128i_i64[1] = -1;
          v24 = 0u;
          v23.m128i_i64[0] = v21;
          v25 = 0;
          v22 = sub_134CB50((__int64)a5, (__int64)&v23, (__int64)a3);
          a1[2] = 1;
          a1[1] = v22;
          *a1 = v22 != 0;
          return a1;
        }
      }
      else
      {
        if ( v20 > 0x75 )
          goto LABEL_14;
        if ( v20 <= 0x72 )
        {
          if ( v20 > 0x70 || v20 == 4 )
            goto LABEL_24;
          goto LABEL_14;
        }
        if ( v20 != 116 )
        {
LABEL_14:
          if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
LABEL_15:
            v15 = _mm_loadu_si128(a3);
            v16 = _mm_loadu_si128(a3 + 1);
            v26 = 1;
            v17 = a3[2].m128i_i64[0];
            v23 = v15;
            v25 = v17;
            v24 = v16;
            v18 = sub_13575E0(a5, v7, &v23, a4);
            a1[2] = 1;
            *a1 = (v18 & 2) != 0;
            a1[1] = (v18 & 4) == 0 ? 3 : 1;
            return a1;
          }
          goto LABEL_12;
        }
      }
LABEL_24:
      a1[2] = 1;
      *(_WORD *)a1 = 0;
      return a1;
    }
  }
  v9 = *(_BYTE *)(v7 + 16);
  if ( v9 == 78 )
  {
    v19 = *(_QWORD *)(v7 - 24);
    if ( *(_BYTE *)(v19 + 16) )
      goto LABEL_15;
    v8 = 0;
    if ( (*(_BYTE *)(v19 + 33) & 0x20) == 0 )
      goto LABEL_15;
    goto LABEL_19;
  }
LABEL_5:
  if ( v6 != 54 || v9 != 54 )
    goto LABEL_15;
  v10 = *(_WORD *)(a4 + 18);
  v11 = *(unsigned __int16 *)(v7 + 18);
  v12 = v11 & v10 & 1;
  if ( !v12 )
    v12 = byte_428C1E0[8 * ((v11 >> 7) & 7) + 4] | (((v10 ^ 0x380) & 0x380) == 0);
  *a1 = v12;
  *(_WORD *)(a1 + 1) = 257;
  return a1;
}
