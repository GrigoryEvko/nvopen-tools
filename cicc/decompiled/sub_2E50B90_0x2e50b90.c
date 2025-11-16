// Function: sub_2E50B90
// Address: 0x2e50b90
//
unsigned __int64 __fastcall sub_2E50B90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  _DWORD *v6; // rdx
  void *v7; // rdx
  __int64 v8; // rsi
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __m128i v11; // xmm0
  int v12; // ecx
  __int64 v13; // r10
  unsigned int v14; // edi
  unsigned __int64 v15; // r8
  __int64 v16; // rdx
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __int64 v20; // rdx
  __m128i v21; // xmm0
  int v22; // ecx
  __int64 v23; // r10
  unsigned int v24; // edi
  unsigned __int64 v25; // r8
  __int64 v26; // rdx
  __m128i *v28; // rdx
  unsigned __int64 result; // rax
  __m128i v30; // xmm0
  unsigned int i; // r15d
  __int64 v33; // rax
  int v34; // edx
  unsigned int v35; // eax
  unsigned int v36; // r8d
  unsigned int v37; // esi
  __int64 v38; // r9
  int v39; // ecx
  unsigned __int64 v40; // rdi
  int v41; // ecx
  __int64 v42; // rdx
  unsigned __int64 v43; // rdi
  unsigned int j; // r15d
  __int64 v48; // rax
  int v49; // edx
  unsigned int v50; // eax
  unsigned int v51; // r9d
  unsigned int v52; // esi
  __int64 v53; // r10
  int v54; // ecx
  unsigned __int64 v55; // r8
  int v56; // ecx
  __int64 v57; // rdx
  unsigned __int64 v58; // r8
  __m128i *v61; // rdx
  __m128i si128; // xmm0
  __int64 v63; // rax
  int v64[14]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  v6 = *(_DWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 3u )
  {
    a2 = sub_CB6200(a2, (unsigned __int8 *)"RP: ", 4u);
  }
  else
  {
    *v6 = 540692562;
    *(_QWORD *)(a2 + 32) += 4LL;
  }
  *(_QWORD *)v64 = *(_QWORD *)a3;
  sub_2D04E80(v64, a2);
  v7 = *(void **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v7 <= 0xCu )
  {
    v8 = sub_CB6200(v4, " Live-in RP: ", 0xDu);
  }
  else
  {
    v8 = v4;
    qmemcpy(v7, " Live-in RP: ", 13);
    *(_QWORD *)(v4 + 32) += 13LL;
  }
  *(_QWORD *)v64 = *(_QWORD *)(a3 + 8);
  sub_2D04E80(v64, v8);
  if ( *(_BYTE *)(a1 + 48) )
  {
    v61 = *(__m128i **)(v4 + 32);
    if ( *(_QWORD *)(v4 + 24) - (_QWORD)v61 <= 0x11u )
    {
      v63 = sub_CB6200(v4, " Register Target: ", 0x12u);
      *(_QWORD *)v64 = *(_QWORD *)(a3 + 16);
      sub_2D04E80(v64, v63);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C0B0);
      v61[1].m128i_i16[0] = 8250;
      *v61 = si128;
      *(_QWORD *)(v4 + 32) += 18LL;
      *(_QWORD *)v64 = *(_QWORD *)(a3 + 16);
      sub_2D04E80(v64, v4);
    }
  }
  v9 = *(_BYTE **)(v4 + 32);
  if ( *(_BYTE **)(v4 + 24) == v9 )
  {
    sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
    v10 = *(_QWORD *)(v4 + 32);
  }
  else
  {
    *v9 = 10;
    v10 = *(_QWORD *)(v4 + 32) + 1LL;
    *(_QWORD *)(v4 + 32) = v10;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v10) <= 0x14 )
  {
    sub_CB6200(v4, "Live-in values begin\n", 0x15u);
  }
  else
  {
    v11 = _mm_load_si128((const __m128i *)&xmmword_3F8C0C0);
    *(_DWORD *)(v10 + 16) = 1852401509;
    *(_BYTE *)(v10 + 20) = 10;
    *(__m128i *)v10 = v11;
    *(_QWORD *)(v4 + 32) += 21LL;
  }
  v12 = *(_DWORD *)(a3 + 88);
  if ( v12 )
  {
    v13 = *(_QWORD *)(a3 + 24);
    v14 = (unsigned int)(v12 - 1) >> 6;
    v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
    v16 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v13 + 8 * v16);
      if ( v14 == (_DWORD)v16 )
        _RCX = v15 & *(_QWORD *)(v13 + 8 * v16);
      if ( _RCX )
        break;
      if ( v14 + 1 == ++v16 )
        goto LABEL_16;
    }
    __asm { tzcnt   rcx, rcx }
    for ( i = ((_DWORD)v16 << 6) + _RCX; i != -1; i = _RAX + ((_DWORD)v42 << 6) )
    {
      v33 = sub_2EBEE10(*(_QWORD *)(a1 + 192), *(unsigned int *)(*(_QWORD *)(a1 + 88) + 4LL * i));
      if ( v33 )
        sub_2E91850(v33, v4, 1, 0, 0, 1, 0);
      v34 = *(_DWORD *)(a3 + 88);
      v35 = i + 1;
      if ( v34 == i + 1 )
        break;
      v36 = v35 >> 6;
      v37 = (unsigned int)(v34 - 1) >> 6;
      if ( v35 >> 6 > v37 )
        break;
      v38 = *(_QWORD *)(a3 + 24);
      v39 = 64 - (v35 & 0x3F);
      v40 = 0xFFFFFFFFFFFFFFFFLL >> v39;
      if ( v39 == 64 )
        v40 = 0;
      v41 = -v34;
      v42 = v36;
      v43 = ~v40;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v38 + 8 * v42);
        if ( v36 == (_DWORD)v42 )
          _RAX = v43 & *(_QWORD *)(v38 + 8 * v42);
        if ( v37 == (_DWORD)v42 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v41;
        if ( _RAX )
          break;
        if ( v37 < (unsigned int)++v42 )
          goto LABEL_16;
      }
      __asm { tzcnt   rax, rax }
    }
  }
LABEL_16:
  v18 = *(__m128i **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v18 <= 0x12u )
  {
    sub_CB6200(v4, "Live-in values end\n", 0x13u);
    v20 = *(_QWORD *)(v4 + 32);
  }
  else
  {
    v19 = _mm_load_si128((const __m128i *)&xmmword_3F8C0D0);
    v18[1].m128i_i8[2] = 10;
    v18[1].m128i_i16[0] = 25710;
    *v18 = v19;
    v20 = *(_QWORD *)(v4 + 32) + 19LL;
    *(_QWORD *)(v4 + 32) = v20;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v20) <= 0x15 )
  {
    sub_CB6200(v4, "Live-out values begin\n", 0x16u);
  }
  else
  {
    v21 = _mm_load_si128((const __m128i *)&xmmword_3F8C0E0);
    *(_DWORD *)(v20 + 16) = 1768383842;
    *(_WORD *)(v20 + 20) = 2670;
    *(__m128i *)v20 = v21;
    *(_QWORD *)(v4 + 32) += 22LL;
  }
  v22 = *(_DWORD *)(a3 + 160);
  if ( v22 )
  {
    v23 = *(_QWORD *)(a3 + 96);
    v24 = (unsigned int)(v22 - 1) >> 6;
    v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
    v26 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v23 + 8 * v26);
      if ( v24 == (_DWORD)v26 )
        _RCX = v25 & *(_QWORD *)(v23 + 8 * v26);
      if ( _RCX )
        break;
      if ( v24 + 1 == ++v26 )
        goto LABEL_26;
    }
    __asm { tzcnt   rcx, rcx }
    for ( j = ((_DWORD)v26 << 6) + _RCX; j != -1; j = ((_DWORD)v57 << 6) + _RAX )
    {
      v48 = sub_2EBEE10(*(_QWORD *)(a1 + 192), *(unsigned int *)(*(_QWORD *)(a1 + 88) + 4LL * j));
      if ( v48 )
        sub_2E91850(v48, v4, 1, 0, 0, 1, 0);
      v49 = *(_DWORD *)(a3 + 160);
      v50 = j + 1;
      if ( v49 == j + 1 )
        break;
      v51 = v50 >> 6;
      v52 = (unsigned int)(v49 - 1) >> 6;
      if ( v50 >> 6 > v52 )
        break;
      v53 = *(_QWORD *)(a3 + 96);
      v54 = 64 - (v50 & 0x3F);
      v55 = 0xFFFFFFFFFFFFFFFFLL >> v54;
      if ( v54 == 64 )
        v55 = 0;
      v56 = -v49;
      v57 = v51;
      v58 = ~v55;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v53 + 8 * v57);
        if ( v51 == (_DWORD)v57 )
          _RAX = v58 & *(_QWORD *)(v53 + 8 * v57);
        if ( v52 == (_DWORD)v57 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> v56;
        if ( _RAX )
          break;
        if ( v52 < (unsigned int)++v57 )
          goto LABEL_26;
      }
      __asm { tzcnt   rax, rax }
    }
  }
LABEL_26:
  v28 = *(__m128i **)(v4 + 32);
  result = *(_QWORD *)(v4 + 24) - (_QWORD)v28;
  if ( result <= 0x13 )
    return sub_CB6200(v4, "Live-out values end\n", 0x14u);
  v30 = _mm_load_si128((const __m128i *)&xmmword_3F8C0E0);
  v28[1].m128i_i32[0] = 174354021;
  *v28 = v30;
  *(_QWORD *)(v4 + 32) += 20LL;
  return result;
}
