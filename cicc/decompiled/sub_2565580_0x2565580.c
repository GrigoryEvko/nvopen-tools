// Function: sub_2565580
// Address: 0x2565580
//
__int64 __fastcall sub_2565580(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
      BUG();
    case 1u:
      v6 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_QWORD *)v3 = off_4A17F98;
      *(_QWORD *)(v3 + 88) = &unk_4A18020;
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
        *(_QWORD *)v3 = off_4A18250;
        *(_WORD *)(v3 + 96) = 256;
        *(_QWORD *)(v3 + 88) = &unk_4A182D8;
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
      *(_QWORD *)v3 = off_4A18338;
      *(_QWORD *)(v3 + 88) = &unk_4A183C0;
      *(_WORD *)(v3 + 96) = 256;
      result = v3;
      break;
    case 4u:
      v9 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v9;
      if ( !v9 )
        goto LABEL_4;
      *(__m128i *)(v9 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v9);
      *(_QWORD *)v3 = off_4A17DC8;
      *(_QWORD *)(v3 + 88) = &unk_4A17E50;
      *(_WORD *)(v3 + 96) = 256;
      result = v3;
      break;
    case 5u:
      v10 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v10;
      if ( !v10 )
        goto LABEL_4;
      *(__m128i *)(v10 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v10);
      *(_QWORD *)v3 = off_4A17EB0;
      *(_QWORD *)(v3 + 88) = &unk_4A17F38;
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
        *(_QWORD *)v3 = off_4A18080;
        *(_WORD *)(v3 + 96) = 256;
        *(_QWORD *)(v3 + 88) = &unk_4A18108;
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
      *(_QWORD *)v3 = off_4A18168;
      *(_QWORD *)(v3 + 88) = &unk_4A181F0;
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
