// Function: sub_89BD20
// Address: 0x89bd20
//
_BOOL8 __fastcall sub_89BD20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _DWORD *a4,
        int a5,
        int a6,
        int a7,
        unsigned __int8 a8)
{
  __m128i *v9; // r15
  __int64 v11; // r13
  int v12; // esi
  unsigned __int8 v13; // r14
  unsigned int v14; // ecx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __m128i *v17; // r13
  unsigned int v18; // esi
  unsigned __int8 v19; // al
  __m128i *v21; // rdx
  const __m128i *v22; // rdi
  __int8 v23; // al
  bool v24; // si
  __int8 v25; // al
  int v26; // [rsp+8h] [rbp-38h]
  bool v28; // [rsp+Ch] [rbp-34h]

  v9 = (__m128i *)a1;
  v11 = *(_QWORD *)(a3 + 88);
  v12 = *(_DWORD *)(a2 + 56);
  v13 = a8;
  v14 = dword_4F077BC;
  if ( dword_4F077BC )
  {
    v14 = 1;
    if ( a6 )
      v14 = qword_4F077A8 <= 0x765Bu;
  }
  if ( v12 )
    v14 |= 2u;
  v26 = sub_89B3C0(**(_QWORD **)(v11 + 32), a1, 1, v14, a4, a8);
  if ( v26 )
  {
    if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0xC3B3u )
      v13 = 5;
    v15 = **(_QWORD ***)(v11 + 32);
    if ( a1 && v15 )
    {
      v16 = (_QWORD *)a1;
      do
      {
        if ( *(_BYTE *)(v15[1] + 80LL) == 3 )
          *(_QWORD *)(v16[8] + 168LL) = *(_QWORD *)(v15[8] + 168LL);
        v16 = (_QWORD *)*v16;
        v15 = (_QWORD *)*v15;
      }
      while ( v16 && v15 );
      v17 = **(__m128i ***)(v11 + 32);
      if ( v17 )
      {
        v28 = a5 == 0;
        while ( 1 )
        {
          v19 = v9[3].m128i_i8[8] & 1;
          if ( v19 )
          {
            if ( v28 )
              break;
          }
          if ( (v17[3].m128i_i8[8] & 1) == 0 || !v19 )
          {
            if ( v19 | v17[3].m128i_i8[8] & 1 )
            {
              if ( v19 )
              {
                v21 = v17;
                v22 = v9;
              }
              else
              {
                v21 = v9;
                v22 = v17;
              }
              v23 = v21[3].m128i_i8[8] | 9;
              v21[3].m128i_i8[8] = v23;
              v24 = (v22[3].m128i_i8[8] & 2) != 0;
              v25 = (2 * v24) | v23 & 0xFD;
              v21[3].m128i_i8[8] = v25;
              v21[3].m128i_i8[8] = v22[3].m128i_i8[8] & 4 | v25 & 0xFB;
              if ( *(_BYTE *)(v9->m128i_i64[1] + 80) == 2 )
                v21[4].m128i_i8[8] = v22[4].m128i_i8[8] & 1 | v21[4].m128i_i8[8] & 0xFE;
              if ( v24 || (v21[3].m128i_i8[8] & 4) != 0 )
              {
                v21[6] = _mm_loadu_si128(v22 + 6);
                v21[7] = _mm_loadu_si128(v22 + 7);
                v21[8].m128i_i64[0] = v22[8].m128i_i64[0];
              }
              v21[5].m128i_i64[0] = v22[5].m128i_i64[0];
            }
            goto LABEL_23;
          }
          sub_684AA0(a8, 0x133u, (_DWORD *)(v9->m128i_i64[1] + 48));
          v9 = (__m128i *)v9->m128i_i64[0];
          v17 = (__m128i *)v17->m128i_i64[0];
          if ( !v9 )
            return v26 != 0;
LABEL_24:
          if ( !v17 )
            return v26 != 0;
        }
        v18 = 953;
        if ( (*(_BYTE *)(a3 + 81) & 0x10) != 0 )
          v18 = (*(_BYTE *)(*(_QWORD *)(a3 + 64) + 177LL) & 0x10) == 0 ? 3273 : 953;
        sub_684AA0(v13, v18, (_DWORD *)(v9->m128i_i64[1] + 48));
LABEL_23:
        v9 = (__m128i *)v9->m128i_i64[0];
        v17 = (__m128i *)v17->m128i_i64[0];
        if ( !v9 )
          return v26 != 0;
        goto LABEL_24;
      }
    }
  }
  return v26 != 0;
}
