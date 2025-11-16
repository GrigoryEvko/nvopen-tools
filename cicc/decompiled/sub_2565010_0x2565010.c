// Function: sub_2565010
// Address: 0x2565010
//
__int64 __fastcall sub_2565010(__m128i *a1, __int64 a2)
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
      v6 = sub_A777F0(0xB8u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_QWORD *)(v3 + 104) = 0;
      *(_QWORD *)v3 = off_4A19650;
      *(_QWORD *)(v3 + 88) = &unk_4A19710;
      *(_QWORD *)(v3 + 136) = v3 + 152;
      *(_QWORD *)(v3 + 144) = 0x400000000LL;
      *(_WORD *)(v3 + 96) = 768;
      *(_QWORD *)(v3 + 112) = 0;
      *(_QWORD *)(v3 + 120) = 0;
      *(_DWORD *)(v3 + 128) = 0;
      result = v3;
      break;
    case 2u:
      v7 = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( !v7 )
        goto LABEL_4;
      *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v7);
      *(_QWORD *)v3 = off_4A19AD0;
      *(_QWORD *)(v3 + 88) = &unk_4A19B90;
      *(_WORD *)(v3 + 96) = 768;
      result = v3;
      break;
    case 3u:
      v8 = sub_A777F0(0xC0u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      *(_QWORD *)(v3 + 104) = 0;
      *(_QWORD *)(v3 + 136) = v3 + 152;
      *(_QWORD *)(v3 + 144) = 0x400000000LL;
      *(_QWORD *)v3 = off_4A199B0;
      *(_QWORD *)(v3 + 88) = &unk_4A19A70;
      *(_WORD *)(v3 + 96) = 768;
      *(_QWORD *)(v3 + 112) = 0;
      *(_QWORD *)(v3 + 120) = 0;
      *(_DWORD *)(v3 + 128) = 0;
      *(_BYTE *)(v3 + 184) = 1;
      result = v3;
      break;
    case 4u:
      v9 = sub_A777F0(0x188u, *(__int64 **)(a2 + 128));
      v3 = v9;
      if ( !v9 )
        goto LABEL_4;
      *(__m128i *)(v9 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v9);
      *(_QWORD *)v3 = off_4A19BF0;
      *(_QWORD *)(v3 + 88) = &unk_4A19CB0;
      *(_QWORD *)(v3 + 136) = v3 + 152;
      *(_QWORD *)(v3 + 144) = 0x800000000LL;
      *(_QWORD *)(v3 + 256) = 0x800000000LL;
      *(_WORD *)(v3 + 96) = 768;
      *(_QWORD *)(v3 + 104) = 0;
      *(_QWORD *)(v3 + 112) = 0;
      *(_QWORD *)(v3 + 120) = 0;
      *(_DWORD *)(v3 + 128) = 0;
      *(_QWORD *)(v3 + 216) = 0;
      *(_QWORD *)(v3 + 224) = 0;
      *(_QWORD *)(v3 + 232) = 0;
      *(_DWORD *)(v3 + 240) = 0;
      *(_QWORD *)(v3 + 248) = v3 + 264;
      *(_QWORD *)(v3 + 328) = 0;
      *(_QWORD *)(v3 + 336) = 0;
      *(_QWORD *)(v3 + 344) = 0;
      *(_DWORD *)(v3 + 352) = 0;
      *(_QWORD *)(v3 + 360) = 0;
      *(_QWORD *)(v3 + 368) = 0;
      *(_QWORD *)(v3 + 376) = 0;
      *(_DWORD *)(v3 + 384) = 0;
      result = v3;
      break;
    case 5u:
      v10 = sub_A777F0(0x188u, *(__int64 **)(a2 + 128));
      v3 = v10;
      if ( !v10 )
        goto LABEL_4;
      *(__m128i *)(v10 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v10);
      *(_QWORD *)(v3 + 136) = v3 + 152;
      *(_QWORD *)(v3 + 144) = 0x800000000LL;
      *(_QWORD *)(v3 + 256) = 0x800000000LL;
      *(_QWORD *)v3 = off_4A19D10;
      *(_QWORD *)(v3 + 88) = &unk_4A19DD0;
      *(_WORD *)(v3 + 96) = 768;
      *(_QWORD *)(v3 + 104) = 0;
      *(_QWORD *)(v3 + 112) = 0;
      *(_QWORD *)(v3 + 120) = 0;
      *(_DWORD *)(v3 + 128) = 0;
      *(_QWORD *)(v3 + 216) = 0;
      *(_QWORD *)(v3 + 224) = 0;
      *(_QWORD *)(v3 + 232) = 0;
      *(_DWORD *)(v3 + 240) = 0;
      *(_QWORD *)(v3 + 248) = v3 + 264;
      *(_QWORD *)(v3 + 328) = 0;
      *(_QWORD *)(v3 + 336) = 0;
      *(_QWORD *)(v3 + 344) = 0;
      *(_DWORD *)(v3 + 352) = 0;
      *(_QWORD *)(v3 + 360) = 0;
      *(_QWORD *)(v3 + 368) = 0;
      *(_QWORD *)(v3 + 376) = 0;
      *(_DWORD *)(v3 + 384) = 0;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0xB8u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        *(_QWORD *)(v3 + 104) = 0;
        *(_QWORD *)(v3 + 136) = v3 + 152;
        *(_QWORD *)(v3 + 144) = 0x400000000LL;
        *(_QWORD *)v3 = off_4A19770;
        *(_WORD *)(v3 + 96) = 768;
        *(_QWORD *)(v3 + 112) = 0;
        *(_QWORD *)(v3 + 120) = 0;
        *(_DWORD *)(v3 + 128) = 0;
        *(_QWORD *)(v3 + 88) = &unk_4A19830;
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
      *(_QWORD *)v3 = off_4A19890;
      *(_QWORD *)(v3 + 88) = &unk_4A19950;
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
