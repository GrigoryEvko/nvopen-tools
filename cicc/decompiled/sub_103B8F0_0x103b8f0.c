// Function: sub_103B8F0
// Address: 0x103b8f0
//
__int64 __fastcall sub_103B8F0(__int64 a1, const __m128i *a2, unsigned __int8 *a3, _QWORD *a4)
{
  __int64 v4; // rbp
  __int64 v6; // rsi
  unsigned __int8 v8; // cl
  int v9; // edi
  __int64 v10; // r8
  unsigned int v11; // r8d
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int16 v18; // dx
  unsigned __int16 v19; // ax
  __m128i v20[3]; // [rsp-48h] [rbp-48h] BYREF
  char v21; // [rsp-18h] [rbp-18h]
  __int64 v22; // [rsp-8h] [rbp-8h]

  v6 = *(_QWORD *)(a1 + 72);
  v8 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 != 85 )
    goto LABEL_2;
  v16 = *(_QWORD *)(v6 - 32);
  if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(v6 + 80) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
    goto LABEL_2;
  v17 = *(_DWORD *)(v16 + 36);
  if ( v17 == 155 )
    return 0;
  if ( v17 <= 0x9B )
  {
    if ( v17 != 11 )
    {
      if ( v17 > 0xB )
      {
        if ( v17 - 69 <= 2 )
          BUG();
      }
      else if ( v17 - 5 <= 1 )
      {
        return 0;
      }
LABEL_2:
      v22 = v4;
      if ( a3 )
      {
        v9 = *a3;
        if ( (unsigned __int8)(v9 - 34) <= 0x33u )
        {
          v10 = 0x8000000000041LL;
          if ( _bittest64(&v10, (unsigned int)(v9 - 34)) )
          {
            LOBYTE(v11) = (unsigned __int8)sub_CF5B00(a4, (unsigned __int8 *)v6, a3) != 0;
            return v11;
          }
          if ( v8 == 61 && (_BYTE)v9 == 61 )
          {
            v18 = *((_WORD *)a3 + 1);
            v19 = *(_WORD *)(v6 + 2);
            v11 = (unsigned __int8)v19 & (unsigned __int8)v18 & 1;
            if ( ((unsigned __int8)v19 & (unsigned __int8)v18 & 1) == 0 )
              LOBYTE(v11) = byte_3F8E4E0[8 * ((v19 >> 7) & 7) + 4] | (((v18 ^ 0x380) & 0x380) == 0);
            return v11;
          }
        }
      }
      v13 = _mm_loadu_si128(a2);
      v21 = 1;
      v14 = _mm_loadu_si128(a2 + 1);
      v15 = _mm_loadu_si128(a2 + 2);
      v20[0] = v13;
      v20[1] = v14;
      v20[2] = v15;
      return (sub_CF6520(a4, (unsigned __int8 *)v6, v20) & 2) != 0;
    }
    return 0;
  }
  if ( v17 > 0xCD )
  {
    if ( v17 != 291 )
      goto LABEL_2;
    return 0;
  }
  else
  {
    if ( v17 <= 0xCB )
      goto LABEL_2;
    return 0;
  }
}
