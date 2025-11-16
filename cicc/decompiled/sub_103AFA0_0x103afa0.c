// Function: sub_103AFA0
// Address: 0x103afa0
//
__int64 __fastcall sub_103AFA0(__int64 a1, const __m128i *a2, unsigned __int8 *a3, _QWORD **a4)
{
  __int64 v4; // rbp
  __int64 v6; // rsi
  unsigned __int8 v8; // cl
  int v9; // edi
  __int64 v10; // r8
  unsigned int v11; // r8d
  __m128i v13; // xmm0
  _QWORD *v14; // rdi
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __int64 v17; // rax
  unsigned int v18; // eax
  unsigned __int16 v19; // dx
  unsigned __int16 v20; // ax
  __m128i v21[3]; // [rsp-48h] [rbp-48h] BYREF
  char v22; // [rsp-18h] [rbp-18h]
  __int64 v23; // [rsp-8h] [rbp-8h]

  v6 = *(_QWORD *)(a1 + 72);
  v8 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 != 85 )
    goto LABEL_2;
  v17 = *(_QWORD *)(v6 - 32);
  if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(v6 + 80) || (*(_BYTE *)(v17 + 33) & 0x20) == 0 )
    goto LABEL_2;
  v18 = *(_DWORD *)(v17 + 36);
  if ( v18 == 155 )
    return 0;
  if ( v18 <= 0x9B )
  {
    if ( v18 != 11 )
    {
      if ( v18 > 0xB )
      {
        if ( v18 - 69 <= 2 )
          BUG();
      }
      else if ( v18 - 5 <= 1 )
      {
        return 0;
      }
LABEL_2:
      v23 = v4;
      if ( a3 )
      {
        v9 = *a3;
        if ( (unsigned __int8)(v9 - 34) <= 0x33u )
        {
          v10 = 0x8000000000041LL;
          if ( _bittest64(&v10, (unsigned int)(v9 - 34)) )
          {
            LOBYTE(v11) = (unsigned __int8)sub_CF5A30(*a4, (unsigned __int8 *)v6, a3, (__int64)(a4 + 1)) != 0;
            return v11;
          }
          if ( v8 == 61 && (_BYTE)v9 == 61 )
          {
            v19 = *((_WORD *)a3 + 1);
            v20 = *(_WORD *)(v6 + 2);
            v11 = (unsigned __int8)v20 & (unsigned __int8)v19 & 1;
            if ( ((unsigned __int8)v20 & (unsigned __int8)v19 & 1) == 0 )
              LOBYTE(v11) = byte_3F8E4E0[8 * ((v20 >> 7) & 7) + 4] | (((v19 ^ 0x380) & 0x380) == 0);
            return v11;
          }
        }
      }
      v13 = _mm_loadu_si128(a2);
      v14 = *a4;
      v15 = _mm_loadu_si128(a2 + 1);
      v16 = _mm_loadu_si128(a2 + 2);
      v22 = 1;
      v21[0] = v13;
      v21[1] = v15;
      v21[2] = v16;
      return (sub_CF63E0(v14, (unsigned __int8 *)v6, v21, (__int64)(a4 + 1)) & 2) != 0;
    }
    return 0;
  }
  if ( v18 > 0xCD )
  {
    if ( v18 != 291 )
      goto LABEL_2;
    return 0;
  }
  else
  {
    if ( v18 <= 0xCB )
      goto LABEL_2;
    return 0;
  }
}
