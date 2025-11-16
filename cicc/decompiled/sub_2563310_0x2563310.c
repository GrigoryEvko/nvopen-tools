// Function: sub_2563310
// Address: 0x2563310
//
__int64 __fastcall sub_2563310(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
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
      v6 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_QWORD *)v3 = off_4A1A870;
      *(_QWORD *)(v3 + 88) = &unk_4A1A8F0;
      *(_WORD *)(v3 + 96) = 256;
      result = v3;
      break;
    case 2u:
      v7 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( v7 )
      {
        *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v7);
        *(_QWORD *)v3 = off_4A1AB10;
        *(_WORD *)(v3 + 96) = 256;
        *(_QWORD *)(v3 + 88) = &unk_4A1AB90;
        BUG();
      }
      goto LABEL_4;
    case 3u:
      v8 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      *(_QWORD *)v3 = off_4A1ABF0;
      *(_QWORD *)(v3 + 88) = &unk_4A1AC70;
      *(_WORD *)(v3 + 96) = 256;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        *(_QWORD *)v3 = off_4A1A950;
        *(_WORD *)(v3 + 96) = 256;
        *(_QWORD *)(v3 + 88) = &unk_4A1A9D0;
      }
      goto LABEL_4;
    case 7u:
      v5 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v5;
      if ( !v5 )
        goto LABEL_4;
      *(__m128i *)(v5 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v5);
      *(_WORD *)(v3 + 96) = 256;
      *(_QWORD *)v3 = off_4A1AA30;
      *(_QWORD *)(v3 + 88) = &unk_4A1AAB0;
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
