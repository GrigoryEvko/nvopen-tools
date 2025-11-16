// Function: sub_2566680
// Address: 0x2566680
//
__int64 __fastcall sub_2566680(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 2u:
      BUG();
    case 1u:
      v6 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_QWORD *)v3 = off_4A1BBA8;
      *(_QWORD *)(v3 + 88) = &unk_4A1BC30;
      *(_WORD *)(v3 + 96) = 768;
      result = v3;
      break;
    case 3u:
      v7 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( !v7 )
        goto LABEL_4;
      *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v7);
      *(_QWORD *)v3 = off_4A1B8F0;
      *(_QWORD *)(v3 + 88) = &unk_4A1B978;
      *(_WORD *)(v3 + 96) = 768;
      result = v3;
      break;
    case 4u:
      v8 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      *(_QWORD *)v3 = off_4A1BAC0;
      *(_QWORD *)(v3 + 88) = &unk_4A1BB48;
      *(_WORD *)(v3 + 96) = 768;
      result = v3;
      break;
    case 5u:
      v9 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v9;
      if ( !v9 )
        goto LABEL_4;
      *(__m128i *)(v9 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v9);
      *(_QWORD *)v3 = off_4A1B9D8;
      *(_QWORD *)(v3 + 88) = &unk_4A1BA60;
      *(_WORD *)(v3 + 96) = 768;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        *(_QWORD *)v3 = off_4A1B720;
        *(_WORD *)(v3 + 96) = 768;
        *(_QWORD *)(v3 + 88) = &unk_4A1B7A8;
      }
      goto LABEL_4;
    case 7u:
      v5 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v5;
      if ( !v5 )
        goto LABEL_4;
      *(__m128i *)(v5 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v5);
      *(_WORD *)(v3 + 96) = 768;
      *(_QWORD *)v3 = off_4A1B808;
      *(_QWORD *)(v3 + 88) = &unk_4A1B890;
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
