// Function: sub_134F8C0
// Address: 0x134f8c0
//
__int64 __fastcall sub_134F8C0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  __int64 result; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm1
  int v13; // eax
  unsigned int v14; // edx
  bool v15; // zf
  __m128i v16; // xmm7
  __m128i v17; // xmm5
  __m128i v18; // [rsp+0h] [rbp-80h] BYREF
  __m128i v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 v20; // [rsp+20h] [rbp-60h]
  __m128i v21; // [rsp+30h] [rbp-50h] BYREF
  __m128i v22; // [rsp+40h] [rbp-40h] BYREF
  __int64 v23; // [rsp+50h] [rbp-30h]

  v5 = a3;
  v6 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v6 > 0x17u )
  {
    if ( (_BYTE)v6 == 78 )
    {
      v8 = a2 | 4;
    }
    else
    {
      if ( (_BYTE)v6 != 29 )
        goto LABEL_4;
      v8 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    }
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      return sub_134F530(a1, v8, a3);
    return 7;
  }
LABEL_4:
  v7 = (unsigned int)(v6 - 24);
  if ( (unsigned int)v7 <= 0x36 )
  {
    a3 = 0x44000200000220LL;
    if ( _bittest64(&a3, v7) )
      return 7;
  }
  switch ( (int)v7 )
  {
    case 30:
      sub_141EB40(&v18, a2, a3, a4, a5);
      goto LABEL_11;
    case 31:
      sub_141EDF0(&v18, a2, a3, a4, a5);
      goto LABEL_18;
    case 34:
      sub_141F110(&v18);
LABEL_11:
      v10 = _mm_loadu_si128(&v18);
      v11 = _mm_loadu_si128(&v19);
      v23 = v20;
      v21 = v10;
      v22 = v11;
      break;
    case 35:
      sub_141F3C0(&v18);
LABEL_18:
      v17 = _mm_loadu_si128(&v19);
      v21 = _mm_loadu_si128(&v18);
      v23 = v20;
      v22 = v17;
      break;
    case 58:
      sub_141F0A0(&v18);
      v16 = _mm_loadu_si128(&v19);
      v21 = _mm_loadu_si128(&v18);
      v23 = v20;
      v22 = v16;
      break;
    default:
      break;
  }
  v12 = _mm_loadu_si128(&v22);
  v18 = _mm_loadu_si128(&v21);
  v20 = v23;
  v19 = v12;
  v13 = sub_134F0E0(a1, v5, (__int64)&v18);
  v14 = v13 | 3;
  v15 = (v13 & 3) == 0;
  result = 4;
  if ( !v15 )
    return v14;
  return result;
}
