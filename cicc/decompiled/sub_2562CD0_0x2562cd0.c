// Function: sub_2562CD0
// Address: 0x2562cd0
//
__int64 __fastcall sub_2562CD0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __m128i v4; // xmm1
  __m128i *v5; // rax
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i v10; // xmm2
  __m128i *v11; // rax
  __int64 v12; // rax
  __m128i v13; // xmm3
  __m128i *v14; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 4u:
    case 5u:
      BUG();
    case 1u:
      v8 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
      v3 = v8;
      if ( !v8 )
        goto LABEL_4;
      *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v8);
      memset((void *)(v3 + 88), 0, 0x58u);
      *(_DWORD *)(v3 + 108) = -1;
      *(_BYTE *)(v3 + 169) = 1;
      *(_QWORD *)(v3 + 96) = &unk_4A16D78;
      *(_QWORD *)(v3 + 136) = v3 + 120;
      *(_QWORD *)(v3 + 144) = v3 + 120;
      *(_QWORD *)(v3 + 160) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A19E30;
      *(_QWORD *)(v3 + 88) = &unk_4A19EB8;
      result = v3;
      break;
    case 2u:
      v9 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
      v3 = v9;
      if ( !v9 )
        goto LABEL_4;
      v10 = _mm_loadu_si128(a1);
      v11 = (__m128i *)(v9 + 56);
      v11[-3].m128i_i64[0] = 0;
      v11[-3].m128i_i64[1] = 0;
      v11[1] = v10;
      v11[-2].m128i_i64[0] = 0;
      v11[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v11;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x58u);
      *(_DWORD *)(v3 + 108) = -1;
      *(_BYTE *)(v3 + 169) = 1;
      *(_QWORD *)(v3 + 96) = &unk_4A16D78;
      *(_QWORD *)(v3 + 136) = v3 + 120;
      *(_QWORD *)(v3 + 144) = v3 + 120;
      *(_QWORD *)(v3 + 160) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A19EF8;
      *(_QWORD *)(v3 + 88) = &unk_4A19F80;
      result = v3;
      break;
    case 3u:
      v12 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
      v3 = v12;
      if ( !v12 )
        goto LABEL_4;
      v13 = _mm_loadu_si128(a1);
      v14 = (__m128i *)(v12 + 56);
      v14[-3].m128i_i64[0] = 0;
      v14[-3].m128i_i64[1] = 0;
      v14[1] = v13;
      v14[-2].m128i_i64[0] = 0;
      v14[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v14;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x58u);
      *(_DWORD *)(v3 + 108) = -1;
      *(_BYTE *)(v3 + 169) = 1;
      *(_QWORD *)(v3 + 96) = &unk_4A16D78;
      *(_QWORD *)(v3 + 136) = v3 + 120;
      *(_QWORD *)(v3 + 144) = v3 + 120;
      *(_QWORD *)(v3 + 160) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A1A150;
      *(_QWORD *)(v3 + 88) = &unk_4A1A1D8;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
      v3 = v2;
      if ( v2 )
      {
        v4 = _mm_loadu_si128(a1);
        v5 = (__m128i *)(v2 + 56);
        v5[-3].m128i_i64[0] = 0;
        v5[-3].m128i_i64[1] = 0;
        v5[1] = v4;
        v5[-2].m128i_i64[0] = 0;
        v5[-2].m128i_i32[2] = 0;
        *(_QWORD *)(v3 + 40) = v5;
        *(_QWORD *)(v3 + 48) = 0x200000000LL;
        memset((void *)(v3 + 88), 0, 0x58u);
        *(_DWORD *)(v3 + 108) = -1;
        *(_BYTE *)(v3 + 169) = 1;
        *(_QWORD *)(v3 + 96) = &unk_4A16D78;
        *(_QWORD *)(v3 + 136) = v3 + 120;
        *(_QWORD *)(v3 + 144) = v3 + 120;
        *(_QWORD *)(v3 + 160) = &unk_4A16CD8;
        *(_QWORD *)v3 = off_4A19FC0;
        *(_QWORD *)(v3 + 88) = &unk_4A1A048;
      }
      goto LABEL_4;
    case 7u:
      v7 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( !v7 )
        goto LABEL_4;
      *(__m128i *)(v7 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v7);
      memset((void *)(v3 + 88), 0, 0x58u);
      *(_DWORD *)(v3 + 108) = -1;
      *(_BYTE *)(v3 + 169) = 1;
      *(_QWORD *)(v3 + 96) = &unk_4A16D78;
      *(_QWORD *)(v3 + 136) = v3 + 120;
      *(_QWORD *)(v3 + 144) = v3 + 120;
      *(_QWORD *)(v3 + 160) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A1A088;
      *(_QWORD *)(v3 + 88) = &unk_4A1A110;
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
