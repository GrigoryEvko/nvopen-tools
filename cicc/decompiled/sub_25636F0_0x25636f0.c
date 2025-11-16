// Function: sub_25636F0
// Address: 0x25636f0
//
__int64 __fastcall sub_25636F0(__m128i *a1, __int64 a2)
{
  __int64 *v2; // rsi
  __int64 v3; // r12
  unsigned int v4; // r13d
  __int64 result; // rax
  __int64 *v6; // rsi
  unsigned int v7; // r13d
  __int64 *v8; // rsi
  unsigned int v9; // r13d
  __int64 *v10; // rsi
  unsigned int v11; // r13d
  __int64 *v12; // rsi
  unsigned int v13; // r13d

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 4u:
    case 5u:
      BUG();
    case 1u:
      v8 = *(__int64 **)(a2 + 128);
      v3 = sub_A777F0(0xB0u, v8);
      if ( !v3 )
        goto LABEL_4;
      v9 = *(_DWORD *)(sub_250D180(a1->m128i_i64, (__int64)v8) + 8);
      *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v3);
      v9 >>= 8;
      *(_DWORD *)(v3 + 96) = v9;
      *(_QWORD *)v3 = &unk_4A16C58;
      *(_QWORD *)(v3 + 88) = &unk_4A16D38;
      sub_AADB10(v3 + 104, v9, 0);
      sub_AADB10(v3 + 136, v9, 1);
      *(_DWORD *)(v3 + 168) = 0;
      *(_QWORD *)v3 = off_4A1C1D8;
      *(_QWORD *)(v3 + 88) = &unk_4A1C268;
      result = v3;
      break;
    case 2u:
      v10 = *(__int64 **)(a2 + 128);
      v3 = sub_A777F0(0xA8u, v10);
      if ( !v3 )
        goto LABEL_4;
      v11 = *(_DWORD *)(sub_250D180(a1->m128i_i64, (__int64)v10) + 8);
      *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v3);
      v11 >>= 8;
      *(_DWORD *)(v3 + 96) = v11;
      *(_QWORD *)v3 = &unk_4A16C58;
      *(_QWORD *)(v3 + 88) = &unk_4A16D38;
      sub_AADB10(v3 + 104, v11, 0);
      sub_AADB10(v3 + 136, v11, 1);
      *(_QWORD *)v3 = off_4A1C108;
      *(_QWORD *)(v3 + 88) = &unk_4A1C198;
      result = v3;
      break;
    case 3u:
      v12 = *(__int64 **)(a2 + 128);
      v3 = sub_A777F0(0xA8u, v12);
      if ( !v3 )
        goto LABEL_4;
      v13 = *(_DWORD *)(sub_250D180(a1->m128i_i64, (__int64)v12) + 8);
      *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v3);
      v13 >>= 8;
      *(_DWORD *)(v3 + 96) = v13;
      *(_QWORD *)v3 = &unk_4A16C58;
      *(_QWORD *)(v3 + 88) = &unk_4A16D38;
      sub_AADB10(v3 + 104, v13, 0);
      sub_AADB10(v3 + 136, v13, 1);
      *(_QWORD *)v3 = off_4A1C2A8;
      *(_QWORD *)(v3 + 88) = &unk_4A1C338;
      result = v3;
      break;
    case 6u:
      v2 = *(__int64 **)(a2 + 128);
      v3 = sub_A777F0(0xA8u, v2);
      if ( v3 )
      {
        v4 = *(_DWORD *)(sub_250D180(a1->m128i_i64, (__int64)v2) + 8);
        *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
        sub_2553350(v3);
        v4 >>= 8;
        *(_DWORD *)(v3 + 96) = v4;
        *(_QWORD *)v3 = &unk_4A16C58;
        *(_QWORD *)(v3 + 88) = &unk_4A16D38;
        sub_AADB10(v3 + 104, v4, 0);
        sub_AADB10(v3 + 136, v4, 1);
        *(_QWORD *)v3 = off_4A1C038;
        *(_QWORD *)(v3 + 88) = &unk_4A1C0C8;
      }
      goto LABEL_4;
    case 7u:
      v6 = *(__int64 **)(a2 + 128);
      v3 = sub_A777F0(0xB0u, v6);
      if ( !v3 )
        goto LABEL_4;
      v7 = *(_DWORD *)(sub_250D180(a1->m128i_i64, (__int64)v6) + 8);
      *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
      sub_2553350(v3);
      v7 >>= 8;
      *(_DWORD *)(v3 + 96) = v7;
      *(_QWORD *)v3 = &unk_4A16C58;
      *(_QWORD *)(v3 + 88) = &unk_4A16D38;
      sub_AADB10(v3 + 104, v7, 0);
      sub_AADB10(v3 + 136, v7, 1);
      *(_DWORD *)(v3 + 168) = 0;
      *(_QWORD *)v3 = off_4A1C378;
      *(_QWORD *)(v3 + 88) = &unk_4A1C408;
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
