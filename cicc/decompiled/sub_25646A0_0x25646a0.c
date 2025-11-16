// Function: sub_25646A0
// Address: 0x25646a0
//
__int64 __fastcall sub_25646A0(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __m128i v4; // xmm1
  __m128i *v5; // rax
  __int64 result; // rax
  __int64 v7; // rax
  __m128i v8; // xmm4
  __m128i *v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i *v12; // rax
  __int64 v13; // rax
  __m128i v14; // xmm2
  __m128i *v15; // rax
  __int64 v16; // rax
  __m128i v17; // xmm3
  __m128i *v18; // rax

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
    case 4u:
    case 5u:
      BUG();
    case 1u:
      v10 = sub_A777F0(0x190u, *(__int64 **)(a2 + 128));
      v3 = v10;
      if ( !v10 )
        goto LABEL_4;
      v11 = _mm_loadu_si128(a1);
      v12 = (__m128i *)(v10 + 56);
      v12[-3].m128i_i64[0] = 0;
      v12[-3].m128i_i64[1] = 0;
      v12[1] = v11;
      v12[-2].m128i_i64[0] = 0;
      v12[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v12;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x138u);
      *(_BYTE *)(v3 + 393) = 1;
      *(_QWORD *)(v3 + 96) = v3 + 112;
      *(_QWORD *)(v3 + 104) = 0x100000000LL;
      *(_QWORD *)(v3 + 288) = v3 + 304;
      *(_QWORD *)(v3 + 296) = 0x400000000LL;
      *(_QWORD *)(v3 + 360) = v3 + 344;
      *(_QWORD *)(v3 + 368) = v3 + 344;
      *(_QWORD *)(v3 + 384) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A17550;
      *(_QWORD *)(v3 + 88) = &unk_4A17608;
      result = v3;
      break;
    case 2u:
      v13 = sub_A777F0(0x190u, *(__int64 **)(a2 + 128));
      v3 = v13;
      if ( !v13 )
        goto LABEL_4;
      v14 = _mm_loadu_si128(a1);
      v15 = (__m128i *)(v13 + 56);
      v15[-3].m128i_i64[0] = 0;
      v15[-3].m128i_i64[1] = 0;
      v15[1] = v14;
      v15[-2].m128i_i64[0] = 0;
      v15[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v15;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x138u);
      *(_BYTE *)(v3 + 393) = 1;
      *(_QWORD *)(v3 + 96) = v3 + 112;
      *(_QWORD *)(v3 + 104) = 0x100000000LL;
      *(_QWORD *)(v3 + 288) = v3 + 304;
      *(_QWORD *)(v3 + 296) = 0x400000000LL;
      *(_QWORD *)(v3 + 360) = v3 + 344;
      *(_QWORD *)(v3 + 368) = v3 + 344;
      *(_QWORD *)(v3 + 384) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A17648;
      *(_QWORD *)(v3 + 88) = &unk_4A17700;
      result = v3;
      break;
    case 3u:
      v16 = sub_A777F0(0x190u, *(__int64 **)(a2 + 128));
      v3 = v16;
      if ( !v16 )
        goto LABEL_4;
      v17 = _mm_loadu_si128(a1);
      v18 = (__m128i *)(v16 + 56);
      v18[-3].m128i_i64[0] = 0;
      v18[-3].m128i_i64[1] = 0;
      v18[1] = v17;
      v18[-2].m128i_i64[0] = 0;
      v18[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v18;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x138u);
      *(_BYTE *)(v3 + 393) = 1;
      *(_QWORD *)(v3 + 96) = v3 + 112;
      *(_QWORD *)(v3 + 104) = 0x100000000LL;
      *(_QWORD *)(v3 + 288) = v3 + 304;
      *(_QWORD *)(v3 + 296) = 0x400000000LL;
      *(_QWORD *)(v3 + 360) = v3 + 344;
      *(_QWORD *)(v3 + 368) = v3 + 344;
      *(_QWORD *)(v3 + 384) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A17930;
      *(_QWORD *)(v3 + 88) = &unk_4A179E8;
      result = v3;
      break;
    case 6u:
      v2 = sub_A777F0(0x190u, *(__int64 **)(a2 + 128));
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
        memset((void *)(v3 + 88), 0, 0x138u);
        *(_BYTE *)(v3 + 393) = 1;
        *(_QWORD *)(v3 + 96) = v3 + 112;
        *(_QWORD *)(v3 + 104) = 0x100000000LL;
        *(_QWORD *)(v3 + 288) = v3 + 304;
        *(_QWORD *)(v3 + 296) = 0x400000000LL;
        *(_QWORD *)(v3 + 360) = v3 + 344;
        *(_QWORD *)(v3 + 368) = v3 + 344;
        *(_QWORD *)(v3 + 384) = &unk_4A16CD8;
        *(_QWORD *)v3 = off_4A17740;
        *(_QWORD *)(v3 + 88) = &unk_4A177F8;
      }
      goto LABEL_4;
    case 7u:
      v7 = sub_A777F0(0x190u, *(__int64 **)(a2 + 128));
      v3 = v7;
      if ( !v7 )
        goto LABEL_4;
      v8 = _mm_loadu_si128(a1);
      v9 = (__m128i *)(v7 + 56);
      v9[-3].m128i_i64[0] = 0;
      v9[-3].m128i_i64[1] = 0;
      v9[1] = v8;
      v9[-2].m128i_i64[0] = 0;
      v9[-2].m128i_i32[2] = 0;
      *(_QWORD *)(v3 + 40) = v9;
      *(_QWORD *)(v3 + 48) = 0x200000000LL;
      memset((void *)(v3 + 88), 0, 0x138u);
      *(_BYTE *)(v3 + 393) = 1;
      *(_QWORD *)(v3 + 96) = v3 + 112;
      *(_QWORD *)(v3 + 104) = 0x100000000LL;
      *(_QWORD *)(v3 + 288) = v3 + 304;
      *(_QWORD *)(v3 + 296) = 0x400000000LL;
      *(_QWORD *)(v3 + 360) = v3 + 344;
      *(_QWORD *)(v3 + 368) = v3 + 344;
      *(_QWORD *)(v3 + 384) = &unk_4A16CD8;
      *(_QWORD *)v3 = off_4A17838;
      *(_QWORD *)(v3 + 88) = &unk_4A178F0;
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
