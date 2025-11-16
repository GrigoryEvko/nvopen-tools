// Function: sub_234B220
// Address: 0x234b220
//
__int64 __fastcall sub_234B220(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __m128i v14; // xmm1
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rdx

  v3 = a1 + 8;
  *(_BYTE *)(v3 - 8) = *(_BYTE *)a2;
  sub_C8CF70(v3, (void *)(a1 + 40), 32, a2 + 40, a2 + 8);
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  v4 = *(_QWORD *)(a2 + 304);
  *(_QWORD *)(a1 + 296) = 1;
  *(_QWORD *)(a1 + 304) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 312);
  ++*(_QWORD *)(a2 + 296);
  *(_DWORD *)(a1 + 312) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 316);
  *(_QWORD *)(a2 + 304) = 0;
  *(_DWORD *)(a1 + 316) = v4;
  LODWORD(v4) = *(_DWORD *)(a2 + 320);
  *(_QWORD *)(a2 + 312) = 0;
  *(_DWORD *)(a1 + 320) = v4;
  v5 = *(_QWORD *)(a2 + 328);
  *(_DWORD *)(a2 + 320) = 0;
  v6 = *(_QWORD *)(a2 + 336);
  *(_QWORD *)(a1 + 328) = v5;
  v7 = *(_QWORD *)(a2 + 344);
  *(_QWORD *)(a1 + 336) = v6;
  *(_QWORD *)(a1 + 344) = v7;
  v8 = *(_QWORD *)(a2 + 352);
  v9 = _mm_loadu_si128((const __m128i *)(a2 + 360));
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 352) = v8;
  *(__m128i *)(a1 + 360) = v9;
  if ( *(_QWORD *)(a2 + 328) == a2 + 376 )
  {
    *(_QWORD *)(a1 + 328) = a1 + 376;
    *(_QWORD *)(a1 + 376) = *(_QWORD *)(a2 + 376);
  }
  if ( v7 )
    *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8 * (*(_QWORD *)(v7 + 8) % v6)) = a1 + 344;
  v10 = *(_QWORD *)(a2 + 384);
  *(_QWORD *)(a2 + 328) = a2 + 376;
  *(_QWORD *)(a2 + 368) = 0;
  v11 = *(_QWORD *)(a2 + 392);
  *(_QWORD *)(a2 + 336) = 1;
  *(_QWORD *)(a2 + 376) = 0;
  *(_QWORD *)(a2 + 344) = 0;
  *(_QWORD *)(a2 + 352) = 0;
  *(_QWORD *)(a1 + 384) = v10;
  v12 = *(_QWORD *)(a2 + 400);
  *(_QWORD *)(a1 + 392) = v11;
  *(_QWORD *)(a1 + 400) = v12;
  v13 = *(_QWORD *)(a2 + 408);
  v14 = _mm_loadu_si128((const __m128i *)(a2 + 416));
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 408) = v13;
  *(__m128i *)(a1 + 416) = v14;
  if ( *(_QWORD *)(a2 + 384) == a2 + 432 )
  {
    *(_QWORD *)(a1 + 384) = a1 + 432;
    *(_QWORD *)(a1 + 432) = *(_QWORD *)(a2 + 432);
  }
  if ( v12 )
    *(_QWORD *)(*(_QWORD *)(a1 + 384) + 8 * (*(_QWORD *)(v12 + 8) % v11)) = a1 + 400;
  *(_QWORD *)(a2 + 384) = a2 + 432;
  *(_QWORD *)(a2 + 424) = 0;
  *(_QWORD *)(a2 + 392) = 1;
  *(_QWORD *)(a2 + 432) = 0;
  *(_QWORD *)(a2 + 400) = 0;
  *(_QWORD *)(a2 + 408) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 464) = 0;
  v15 = *(_QWORD *)(a2 + 448);
  v16 = *(_DWORD *)(a2 + 464);
  ++*(_QWORD *)(a2 + 440);
  *(_QWORD *)(a1 + 448) = v15;
  v17 = *(_QWORD *)(a2 + 456);
  *(_QWORD *)(a1 + 440) = 1;
  *(_QWORD *)(a1 + 456) = v17;
  *(_DWORD *)(a1 + 464) = v16;
  *(_QWORD *)(a2 + 448) = 0;
  *(_QWORD *)(a2 + 456) = 0;
  *(_DWORD *)(a2 + 464) = 0;
  return sub_C8CF70(a1 + 472, (void *)(a1 + 504), 32, a2 + 504, a2 + 472);
}
