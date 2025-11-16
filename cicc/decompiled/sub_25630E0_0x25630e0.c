// Function: sub_25630E0
// Address: 0x25630e0
//
_QWORD *__fastcall sub_25630E0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *result; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 4u:
    case 5u:
      BUG();
    case 1u:
      v6 = sub_A777F0(0x70u, *(__int64 **)(a2 + 128));
      v3 = (_QWORD *)v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      v3[12] = 1;
      v3[13] = 0x100000000LL;
      *v3 = off_4A1A218;
      v3[11] = &unk_4A1A2A0;
      result = v3;
      break;
    case 2u:
      v7 = sub_A777F0(0x70u, *(__int64 **)(a2 + 128));
      v3 = (_QWORD *)v7;
      if ( !v7 )
        goto LABEL_4;
      *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v7);
      v3[12] = 1;
      v3[13] = 0x100000000LL;
      *v3 = off_4A1A300;
      v3[11] = &unk_4A1A388;
      result = v3;
      break;
    case 3u:
      v8 = sub_A777F0(0x70u, *(__int64 **)(a2 + 128));
      v3 = (_QWORD *)v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      v3[12] = 1;
      v3[13] = 0x100000000LL;
      *v3 = off_4A1A5B8;
      v3[11] = &unk_4A1A640;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x70u, *(__int64 **)(a2 + 128));
      v3 = (_QWORD *)v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        v3[12] = 1;
        v3[13] = 0x100000000LL;
        *v3 = off_4A1A3E8;
        v3[11] = &unk_4A1A470;
      }
      goto LABEL_4;
    case 7u:
      v5 = sub_A777F0(0x70u, *(__int64 **)(a2 + 128));
      v3 = (_QWORD *)v5;
      if ( !v5 )
        goto LABEL_4;
      *(__m128i *)(v5 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v5);
      v3[12] = 1;
      v3[13] = 0x100000000LL;
      *v3 = off_4A1A4D0;
      v3[11] = &unk_4A1A558;
      result = v3;
      break;
    default:
      v3 = 0;
LABEL_4:
      result = v3;
      break;
  }
  return result;
}
