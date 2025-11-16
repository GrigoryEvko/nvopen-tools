// Function: sub_2BEEC60
// Address: 0x2beec60
//
__int64 *__fastcall sub_2BEEC60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12[3]; // [rsp+8h] [rbp-18h] BYREF

  switch ( a3 )
  {
    case 4LL:
      if ( *(_DWORD *)a2 != 1819047278 )
      {
LABEL_3:
        *a1 = 0;
        return a1;
      }
      v3 = sub_22077B0(0x28u);
      if ( v3 )
      {
        strcpy((char *)(v3 + 24), "null");
        *(_QWORD *)(v3 + 8) = v3 + 24;
        *(_QWORD *)(v3 + 16) = 4;
        *(_QWORD *)v3 = &unk_4A238E0;
      }
LABEL_19:
      *a1 = v3;
      return a1;
    case 23LL:
      if ( *(_QWORD *)a2 ^ 0x6E692D746E697270LL | *(_QWORD *)(a2 + 8) ^ 0x6F69746375727473LL
        || *(_DWORD *)(a2 + 16) != 1868770670
        || *(_WORD *)(a2 + 20) != 28277
        || *(_BYTE *)(a2 + 22) != 116 )
      {
        goto LABEL_3;
      }
      v3 = sub_22077B0(0x28u);
      if ( v3 )
      {
        strcpy((char *)(v3 + 24), "null");
        *(_QWORD *)(v3 + 8) = v3 + 24;
        *(_QWORD *)(v3 + 16) = 4;
        *(_QWORD *)v3 = &unk_4A23910;
      }
      goto LABEL_19;
    case 7LL:
      if ( *(_DWORD *)a2 != 1932358260 || *(_WORD *)(a2 + 4) != 30305 || *(_BYTE *)(a2 + 6) != 101 )
        goto LABEL_3;
      v3 = sub_22077B0(0x28u);
      if ( v3 )
      {
        *(_BYTE *)(v3 + 30) = 101;
        *(_QWORD *)(v3 + 8) = v3 + 24;
        *(_DWORD *)(v3 + 24) = 1932358260;
        *(_WORD *)(v3 + 28) = 30305;
        *(_QWORD *)(v3 + 16) = 7;
        *(_BYTE *)(v3 + 31) = 0;
        *(_QWORD *)v3 = &unk_4A349A0;
      }
      goto LABEL_19;
    case 9LL:
      if ( *(_QWORD *)a2 != 0x70656363612D7274LL || *(_BYTE *)(a2 + 8) != 116 )
        goto LABEL_3;
      v3 = sub_22077B0(0x28u);
      if ( v3 )
      {
        strcpy((char *)(v3 + 24), "tr-accept");
        *(_QWORD *)(v3 + 8) = v3 + 24;
        *(_QWORD *)(v3 + 16) = 9;
        *(_QWORD *)v3 = &unk_4A23940;
      }
      goto LABEL_19;
  }
  if ( a3 != 19 )
  {
    if ( a3 != 13
      || *(_QWORD *)a2 != 0x752D6D6F74746F62LL
      || *(_DWORD *)(a2 + 8) != 1702243696
      || *(_BYTE *)(a2 + 12) != 99 )
    {
      goto LABEL_3;
    }
    v3 = sub_22077B0(0x100u);
    if ( v3 )
    {
      *(_QWORD *)(v3 + 8) = v3 + 24;
      strcpy((char *)(v3 + 24), "bottom-up-vec");
      *(_QWORD *)v3 = &unk_4A348B0;
      *(_QWORD *)(v3 + 16) = 13;
      *(_BYTE *)(v3 + 40) = 0;
      *(_QWORD *)(v3 + 48) = 0;
      *(_QWORD *)(v3 + 56) = 0;
      *(_QWORD *)(v3 + 64) = 0;
      *(_QWORD *)(v3 + 72) = 0;
      *(_DWORD *)(v3 + 80) = 0;
      *(_QWORD *)(v3 + 88) = 0;
      *(_QWORD *)(v3 + 96) = 0;
      *(_QWORD *)(v3 + 104) = v3 + 120;
      *(_QWORD *)(v3 + 112) = 0x1000000000LL;
      *(_DWORD *)(v3 + 248) = 0;
    }
    goto LABEL_19;
  }
  if ( *(_QWORD *)a2 ^ 0x70656363612D7274LL | *(_QWORD *)(a2 + 8) ^ 0x7665722D726F2D74LL
    || *(_WORD *)(a2 + 16) != 29285
    || *(_BYTE *)(a2 + 18) != 116 )
  {
    goto LABEL_3;
  }
  v5 = (_QWORD *)sub_22077B0(0x28u);
  v6 = v5;
  if ( v5 )
  {
    v12[0] = 19;
    *v5 = &unk_4A23850;
    v5[1] = v5 + 3;
    v7 = sub_22409D0((__int64)(v5 + 1), v12, 0);
    v8 = v12[0];
    si128 = _mm_load_si128((const __m128i *)&xmmword_43A0150);
    v6[1] = v7;
    v6[3] = v8;
    *(_WORD *)(v7 + 16) = 29285;
    *(_BYTE *)(v7 + 18) = 116;
    *(__m128i *)v7 = si128;
    v10 = v12[0];
    v11 = v6[1];
    v6[2] = v12[0];
    *(_BYTE *)(v11 + v10) = 0;
    *v6 = &unk_4A34970;
  }
  *a1 = (__int64)v6;
  return a1;
}
