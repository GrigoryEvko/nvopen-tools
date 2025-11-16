// Function: sub_2563EA0
// Address: 0x2563ea0
//
__int64 __fastcall sub_2563EA0(__m128i *a1, __int64 a2)
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
      v6 = sub_A777F0(0x168u, *(__int64 **)(a2 + 128));
      v3 = v6;
      if ( !v6 )
        goto LABEL_4;
      *(__m128i *)(v6 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v6);
      *(_QWORD *)(v3 + 112) = 0;
      *(_WORD *)(v3 + 104) = 256;
      *(_QWORD *)(v3 + 120) = 0;
      *(_QWORD *)(v3 + 96) = &unk_4A16CD8;
      *(_QWORD *)(v3 + 144) = v3 + 160;
      *(_QWORD *)(v3 + 152) = 0x800000000LL;
      *(_QWORD *)v3 = off_4A1D6A0;
      *(_QWORD *)(v3 + 88) = &unk_4A1D738;
      *(_QWORD *)(v3 + 128) = 0;
      *(_DWORD *)(v3 + 136) = 0;
      *(_BYTE *)(v3 + 352) = 0;
      result = v3;
      break;
    case 2u:
      v7 = sub_A777F0(0x170u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( !v7 )
        goto LABEL_4;
      *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v7);
      *(_QWORD *)(v3 + 112) = 0;
      *(_WORD *)(v3 + 104) = 256;
      *(_QWORD *)(v3 + 120) = 0;
      *(_QWORD *)(v3 + 96) = &unk_4A16CD8;
      *(_QWORD *)(v3 + 144) = v3 + 160;
      *(_QWORD *)(v3 + 152) = 0x800000000LL;
      *(_QWORD *)v3 = off_4A1D850;
      *(_QWORD *)(v3 + 88) = &unk_4A1D8E8;
      *(_QWORD *)(v3 + 128) = 0;
      *(_DWORD *)(v3 + 136) = 0;
      *(_BYTE *)(v3 + 352) = 0;
      *(_QWORD *)(v3 + 360) = 0;
      result = v3;
      break;
    case 3u:
      v8 = sub_A777F0(0x168u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      *(_QWORD *)(v3 + 112) = 0;
      *(_WORD *)(v3 + 104) = 256;
      *(_QWORD *)(v3 + 120) = 0;
      *(_QWORD *)(v3 + 96) = &unk_4A16CD8;
      *(_QWORD *)(v3 + 144) = v3 + 160;
      *(_QWORD *)(v3 + 152) = 0x800000000LL;
      *(_QWORD *)v3 = off_4A1D928;
      *(_QWORD *)(v3 + 88) = &unk_4A1D9C0;
      *(_QWORD *)(v3 + 128) = 0;
      *(_DWORD *)(v3 + 136) = 0;
      *(_BYTE *)(v3 + 352) = 0;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x168u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        *(__m128i *)(v2 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v2);
        *(_QWORD *)(v3 + 112) = 0;
        *(_WORD *)(v3 + 104) = 256;
        *(_QWORD *)(v3 + 120) = 0;
        *(_QWORD *)(v3 + 96) = &unk_4A16CD8;
        *(_QWORD *)(v3 + 144) = v3 + 160;
        *(_QWORD *)(v3 + 152) = 0x800000000LL;
        *(_QWORD *)v3 = off_4A1D778;
        *(_QWORD *)(v3 + 128) = 0;
        *(_DWORD *)(v3 + 136) = 0;
        *(_BYTE *)(v3 + 352) = 0;
        *(_QWORD *)(v3 + 88) = &unk_4A1D810;
      }
      goto LABEL_4;
    case 7u:
      v5 = sub_A777F0(0x168u, *(__int64 **)(a2 + 128));
      v3 = v5;
      if ( !v5 )
        goto LABEL_4;
      *(__m128i *)(v5 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v5);
      *(_QWORD *)(v3 + 112) = 0;
      *(_WORD *)(v3 + 104) = 256;
      *(_QWORD *)(v3 + 120) = 0;
      *(_QWORD *)(v3 + 128) = 0;
      *(_QWORD *)(v3 + 96) = &unk_4A16CD8;
      *(_QWORD *)(v3 + 144) = v3 + 160;
      *(_QWORD *)(v3 + 152) = 0x800000000LL;
      *(_QWORD *)v3 = off_4A1DA00;
      *(_QWORD *)(v3 + 88) = &unk_4A1DA98;
      *(_DWORD *)(v3 + 136) = 0;
      *(_BYTE *)(v3 + 352) = 0;
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
