// Function: sub_69BFB0
// Address: 0x69bfb0
//
__int64 __fastcall sub_69BFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rax
  char i; // dl
  unsigned int v11; // r12d
  __m128i v13; // xmm2
  __int64 v14; // rdx
  __m128i *v15; // rax
  __m128i v16; // xmm3
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __int64 v23; // rsi
  __int64 v24; // rax
  char j; // dl
  __int64 v26; // rax
  char k; // dl
  int v28; // [rsp+4h] [rbp-4FCh] BYREF
  __int64 v29; // [rsp+8h] [rbp-4F8h] BYREF
  __m128i v30; // [rsp+10h] [rbp-4F0h] BYREF
  __m128i v31; // [rsp+20h] [rbp-4E0h] BYREF
  __m128i v32; // [rsp+30h] [rbp-4D0h] BYREF
  __m128i v33; // [rsp+40h] [rbp-4C0h] BYREF
  __m128i v34; // [rsp+50h] [rbp-4B0h] BYREF
  __m128i v35; // [rsp+60h] [rbp-4A0h] BYREF
  __m128i v36; // [rsp+70h] [rbp-490h] BYREF
  __m128i v37; // [rsp+80h] [rbp-480h] BYREF
  __m128i v38; // [rsp+90h] [rbp-470h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-460h]
  __int64 v40[44]; // [rsp+B0h] [rbp-450h] BYREF
  __int64 v41[44]; // [rsp+210h] [rbp-2F0h] BYREF
  _QWORD v42[2]; // [rsp+370h] [rbp-190h] BYREF
  char v43; // [rsp+380h] [rbp-180h]

  v6 = a2;
  v7 = *(unsigned __int8 *)(a1 + 140);
  if ( (unsigned __int8)(v7 - 9) <= 2u )
    goto LABEL_5;
  if ( (_BYTE)v7 == 2 )
  {
    if ( (*(_BYTE *)(a1 + 161) & 8) == 0 )
      return 0;
LABEL_5:
    v28 = 0;
    v29 = sub_724DC0(a1, a2, (unsigned int)(v7 - 9), a4, a5, a6);
    v30 = _mm_loadu_si128((const __m128i *)qword_4D03C50);
    v31 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 16LL));
    v8 = *(_QWORD *)(qword_4D03C50 + 144LL);
    v32 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 32LL));
    v33 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 48LL));
    v39 = v8;
    v34 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 64LL));
    v35 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 80LL));
    v36 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 96LL));
    v37 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 112LL));
    v38 = _mm_loadu_si128((const __m128i *)(qword_4D03C50 + 128LL));
    sub_68B920(a1, v29, (__int64)v40, (__int64)v41);
    sub_68FEF0(v40, v41, &dword_4F063F8, dword_4F06650[0], (__int64)&v28, (__int64)v42);
    if ( v43 )
    {
      v9 = v42[0];
      for ( i = *(_BYTE *)(v42[0] + 140LL); i == 12; i = *(_BYTE *)(v9 + 140) )
        v9 = *(_QWORD *)(v9 + 160);
      if ( i && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
        goto LABEL_17;
    }
    if ( v28 )
    {
      v13 = _mm_loadu_si128(&v31);
      v14 = v39;
      v15 = (__m128i *)qword_4D03C50;
      v16 = _mm_loadu_si128(&v32);
      v17 = _mm_loadu_si128(&v33);
      v18 = _mm_loadu_si128(&v34);
      *(__m128i *)qword_4D03C50 = _mm_loadu_si128(&v30);
      v19 = _mm_loadu_si128(&v35);
      v20 = _mm_loadu_si128(&v36);
      v15[9].m128i_i64[0] = v14;
      v21 = _mm_loadu_si128(&v37);
      v15[1] = v13;
      v22 = _mm_loadu_si128(&v38);
      v23 = v29;
      v15[2] = v16;
      v15[3] = v17;
      v15[4] = v18;
      v15[5] = v19;
      v15[6] = v20;
      v15[7] = v21;
      v15[8] = v22;
      sub_68B920(a1, v23, (__int64)v40, (__int64)v41);
      sub_6907F0(v40, v41, 0x2Fu, dword_4F07508, dword_4F06650[0], (__int64)v42);
      if ( v43 )
      {
        v24 = v42[0];
        for ( j = *(_BYTE *)(v42[0] + 140LL); j == 12; j = *(_BYTE *)(v24 + 140) )
          v24 = *(_QWORD *)(v24 + 160);
        if ( j && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
        {
          if ( (unsigned int)(v6 - 1) > 1 && v6 != 4 )
          {
LABEL_17:
            v11 = 0;
            goto LABEL_11;
          }
          sub_6E4710(v42);
          sub_68B920(a1, v29, (__int64)v40, (__int64)v41);
          sub_69B310(v40, v41, 0x2Bu, dword_4F07508, dword_4F06650[0], (__int64)v42);
          if ( v43 )
          {
            v26 = v42[0];
            for ( k = *(_BYTE *)(v42[0] + 140LL); k == 12; k = *(_BYTE *)(v26 + 140) )
              v26 = *(_QWORD *)(v26 + 160);
            if ( k )
            {
              v11 = *(_BYTE *)(qword_4D03C50 + 19LL) & 1;
              goto LABEL_11;
            }
          }
        }
      }
    }
    v11 = 1;
LABEL_11:
    sub_6E4710(v42);
    sub_724E30(&v29);
    return v11;
  }
  switch ( (char)v7 )
  {
    case 3:
      v11 = (unsigned int)(a2 - 1) <= 1;
      break;
    case 6:
      if ( !(unsigned int)sub_8D2340(a1) )
        return 0;
      goto LABEL_18;
    case 13:
    case 19:
LABEL_18:
      v11 = (_DWORD)a2 == 4 || (unsigned int)(a2 - 1) <= 1;
      break;
    default:
      v11 = 1;
      break;
  }
  return v11;
}
