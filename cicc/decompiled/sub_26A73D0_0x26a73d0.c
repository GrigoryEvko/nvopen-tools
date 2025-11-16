// Function: sub_26A73D0
// Address: 0x26a73d0
//
__int64 __fastcall sub_26A73D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6)
{
  int v9; // ecx
  int v10; // ecx
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // eax
  int v18; // edx
  int v19; // r8d
  unsigned int v20; // eax
  void *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rbx
  __m128i v30; // xmm2
  __m128i *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  int v37; // r15d
  __int64 v39; // rax
  __m128i v40; // xmm3
  __m128i *v41; // rax
  char v42; // al
  __int64 v43; // [rsp+8h] [rbp-78h]
  char v44; // [rsp+10h] [rbp-70h]
  __m128i v46; // [rsp+20h] [rbp-60h] BYREF
  void *v47; // [rsp+30h] [rbp-50h] BYREF
  __m128i v48; // [rsp+38h] [rbp-48h]

  v46.m128i_i64[0] = a2;
  v46.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v46) )
    v46.m128i_i64[1] = 0;
  v9 = *(_DWORD *)(a1 + 160);
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = 1;
    for ( i = v10
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned __int64)(((unsigned int)&unk_438FC88 >> 9) ^ ((unsigned int)&unk_438FC88 >> 4)) << 32)
                | (16 * (((unsigned __int32)v46.m128i_i32[0] >> 9) ^ ((unsigned __int32)v46.m128i_i32[0] >> 4)))
                ^ ((unsigned __int32)v46.m128i_i32[2] >> 4)
                ^ ((unsigned __int32)v46.m128i_i32[2] >> 9))) >> 31)
             ^ (484763065
              * ((16 * (((unsigned __int32)v46.m128i_i32[0] >> 9) ^ ((unsigned __int32)v46.m128i_i32[0] >> 4)))
               ^ ((unsigned __int32)v46.m128i_i32[2] >> 4)
               ^ ((unsigned __int32)v46.m128i_i32[2] >> 9)))); ; i = v10 & v14 )
    {
      v13 = *(_QWORD *)(a1 + 144) + 32LL * i;
      if ( *(_UNKNOWN **)v13 == &unk_438FC88 && *(_OWORD *)&v46 == *(_OWORD *)(v13 + 8) )
        break;
      if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == qword_4FEE4D0 && *(_QWORD *)(v13 + 16) == qword_4FEE4D8 )
        goto LABEL_10;
      v14 = v11 + i;
      ++v11;
    }
    v29 = *(_QWORD *)(v13 + 24);
    if ( v29 )
    {
      if ( a5 == 2 )
        return v29;
      goto LABEL_39;
    }
  }
LABEL_10:
  v15 = *(_QWORD *)(a1 + 4376);
  if ( v15 )
  {
    v16 = *(_QWORD *)(v15 + 8);
    v17 = *(_DWORD *)(v15 + 24);
    if ( !v17 )
      return 0;
    v18 = v17 - 1;
    v19 = 1;
    v20 = (v17 - 1) & (((unsigned int)&unk_438FC88 >> 9) ^ ((unsigned int)&unk_438FC88 >> 4));
    v21 = *(void **)(v16 + 8LL * v20);
    if ( v21 != &unk_438FC88 )
    {
      while ( v21 != (void *)-4096LL )
      {
        v20 = v18 & (v19 + v20);
        v21 = *(void **)(v16 + 8LL * v20);
        if ( v21 == &unk_438FC88 )
          goto LABEL_13;
        ++v19;
      }
      return 0;
    }
  }
LABEL_13:
  v22 = sub_25096F0(&v46);
  v23 = v22;
  if ( v22 && ((unsigned __int8)sub_B2D610(v22, 20) || (unsigned __int8)sub_B2D610(v23, 48))
    || *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
  {
    return 0;
  }
  v44 = sub_2673B80(a1, v46.m128i_i64);
  v24 = v46.m128i_i8[0] & 3;
  if ( v24 == 3 )
    goto LABEL_53;
  if ( v24 == 2 )
    goto LABEL_53;
  if ( (v46.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL) == 0 )
    goto LABEL_53;
  v25 = *(unsigned __int8 *)(v46.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL);
  if ( (_BYTE)v25 == 22 )
    goto LABEL_53;
  if ( !(_BYTE)v25 )
  {
    if ( (_BYTE)v24 != 1 )
    {
      v39 = sub_A777F0(0x218u, *(__int64 **)(a1 + 128));
      v29 = v39;
      if ( v39 )
      {
        v40 = _mm_loadu_si128(&v46);
        v41 = (__m128i *)(v39 + 56);
        v41[-3].m128i_i64[0] = 0;
        v41[-3].m128i_i64[1] = 0;
        v41[1] = v40;
        v41[-2].m128i_i64[0] = 0;
        v41[-2].m128i_i32[2] = 0;
        *(_QWORD *)(v29 + 40) = v41;
        *(_QWORD *)(v29 + 48) = 0x200000000LL;
        *(_QWORD *)(v29 + 104) = off_4A1FB78;
        *(_QWORD *)(v29 + 152) = v29 + 168;
        *(_QWORD *)(v29 + 168) = off_4A1FBD8;
        *(_QWORD *)(v29 + 216) = v29 + 232;
        *(_QWORD *)(v29 + 232) = off_4A1FC38;
        *(_QWORD *)(v29 + 280) = v29 + 296;
        *(_BYTE *)(v29 + 96) = 0;
        *(_WORD *)(v29 + 112) = 256;
        *(_QWORD *)(v29 + 120) = 0;
        *(_QWORD *)(v29 + 128) = 0;
        *(_QWORD *)(v29 + 136) = 0;
        *(_DWORD *)(v29 + 144) = 0;
        *(_QWORD *)(v29 + 160) = 0;
        *(_WORD *)(v29 + 176) = 256;
        *(_QWORD *)(v29 + 184) = 0;
        *(_QWORD *)(v29 + 192) = 0;
        *(_QWORD *)(v29 + 200) = 0;
        *(_DWORD *)(v29 + 208) = 0;
        *(_QWORD *)(v29 + 224) = 0;
        *(_WORD *)(v29 + 240) = 256;
        *(_QWORD *)(v29 + 248) = 0;
        *(_QWORD *)(v29 + 256) = 0;
        *(_QWORD *)(v29 + 264) = 0;
        *(_DWORD *)(v29 + 272) = 0;
        *(_QWORD *)(v29 + 288) = 0;
        *(_QWORD *)(v29 + 328) = off_4A1FC98;
        *(_QWORD *)(v29 + 376) = v29 + 392;
        *(_QWORD *)(v29 + 392) = off_4A1FCF8;
        *(_QWORD *)(v29 + 440) = v29 + 464;
        *(_QWORD *)v29 = off_4A203D8;
        *(_QWORD *)(v29 + 88) = &unk_4A20458;
        *(_QWORD *)(v29 + 296) = 0;
        *(_QWORD *)(v29 + 304) = 0;
        *(_QWORD *)(v29 + 312) = 0;
        *(_BYTE *)(v29 + 320) = 0;
        *(_WORD *)(v29 + 336) = 256;
        *(_QWORD *)(v29 + 344) = 0;
        *(_QWORD *)(v29 + 352) = 0;
        *(_QWORD *)(v29 + 360) = 0;
        *(_DWORD *)(v29 + 368) = 0;
        *(_QWORD *)(v29 + 384) = 0;
        *(_WORD *)(v29 + 400) = 256;
        *(_QWORD *)(v29 + 408) = 0;
        *(_QWORD *)(v29 + 416) = 0;
        *(_QWORD *)(v29 + 424) = 0;
        *(_DWORD *)(v29 + 432) = 0;
        *(_QWORD *)(v29 + 448) = 0;
        *(_QWORD *)(v29 + 456) = 0;
        *(_BYTE *)(v29 + 464) = 0;
        *(_QWORD *)(v29 + 472) = 0;
        *(_QWORD *)(v29 + 480) = v29 + 504;
        *(_QWORD *)(v29 + 488) = 4;
        *(_DWORD *)(v29 + 496) = 0;
        *(_BYTE *)(v29 + 500) = 1;
        goto LABEL_28;
      }
LABEL_54:
      v47 = &unk_438FC88;
      BUG();
    }
LABEL_53:
    BUG();
  }
  if ( (unsigned __int8)v25 <= 0x1Cu )
    goto LABEL_53;
  v26 = (unsigned int)(v25 - 34);
  if ( (unsigned __int8)v26 > 0x33u )
    goto LABEL_53;
  v27 = 0x8000000000041LL;
  if ( !_bittest64(&v27, v26) || (_BYTE)v24 == 1 )
    goto LABEL_53;
  v28 = sub_A777F0(0x1D8u, *(__int64 **)(a1 + 128));
  v29 = v28;
  if ( !v28 )
    goto LABEL_54;
  v30 = _mm_loadu_si128(&v46);
  *(_QWORD *)(v28 + 16) = 0;
  v31 = (__m128i *)(v28 + 56);
  v31[-3].m128i_i64[0] = 0;
  v31[1] = v30;
  v31[-2].m128i_i64[0] = 0;
  v31[-2].m128i_i32[2] = 0;
  *(_QWORD *)(v29 + 40) = v31;
  *(_QWORD *)(v29 + 48) = 0x200000000LL;
  *(_QWORD *)(v29 + 104) = off_4A1FB78;
  *(_QWORD *)(v29 + 152) = v29 + 168;
  *(_QWORD *)(v29 + 168) = off_4A1FBD8;
  *(_QWORD *)(v29 + 216) = v29 + 232;
  *(_QWORD *)(v29 + 232) = off_4A1FC38;
  *(_QWORD *)(v29 + 280) = v29 + 296;
  *(_BYTE *)(v29 + 96) = 0;
  *(_WORD *)(v29 + 112) = 256;
  *(_QWORD *)(v29 + 120) = 0;
  *(_QWORD *)(v29 + 128) = 0;
  *(_QWORD *)(v29 + 136) = 0;
  *(_DWORD *)(v29 + 144) = 0;
  *(_QWORD *)(v29 + 160) = 0;
  *(_WORD *)(v29 + 176) = 256;
  *(_QWORD *)(v29 + 184) = 0;
  *(_QWORD *)(v29 + 192) = 0;
  *(_QWORD *)(v29 + 200) = 0;
  *(_DWORD *)(v29 + 208) = 0;
  *(_QWORD *)(v29 + 224) = 0;
  *(_WORD *)(v29 + 240) = 256;
  *(_QWORD *)(v29 + 248) = 0;
  *(_QWORD *)(v29 + 256) = 0;
  *(_QWORD *)(v29 + 264) = 0;
  *(_DWORD *)(v29 + 272) = 0;
  *(_QWORD *)(v29 + 288) = 0;
  *(_QWORD *)(v29 + 328) = off_4A1FC98;
  *(_QWORD *)(v29 + 376) = v29 + 392;
  *(_WORD *)(v29 + 400) = 256;
  *(_QWORD *)(v29 + 392) = off_4A1FCF8;
  *(_QWORD *)(v29 + 440) = v29 + 464;
  *(_QWORD *)v29 = off_4A20498;
  *(_QWORD *)(v29 + 296) = 0;
  *(_QWORD *)(v29 + 304) = 0;
  *(_QWORD *)(v29 + 312) = 0;
  *(_BYTE *)(v29 + 320) = 0;
  *(_WORD *)(v29 + 336) = 256;
  *(_QWORD *)(v29 + 344) = 0;
  *(_QWORD *)(v29 + 352) = 0;
  *(_QWORD *)(v29 + 360) = 0;
  *(_DWORD *)(v29 + 368) = 0;
  *(_QWORD *)(v29 + 384) = 0;
  *(_QWORD *)(v29 + 408) = 0;
  *(_QWORD *)(v29 + 416) = 0;
  *(_QWORD *)(v29 + 424) = 0;
  *(_DWORD *)(v29 + 432) = 0;
  *(_QWORD *)(v29 + 448) = 0;
  *(_QWORD *)(v29 + 456) = 0;
  *(_BYTE *)(v29 + 464) = 0;
  *(_QWORD *)(v29 + 88) = &unk_4A20518;
LABEL_28:
  v47 = &unk_438FC88;
  v48 = _mm_loadu_si128((const __m128i *)(v29 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v47) = v29;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v47 = (void *)(v29 & 0xFFFFFFFFFFFFFFFBLL);
    sub_269CF50(a1 + 224, (unsigned __int64 *)&v47, v32, v33, v34, v35);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v29) )
      goto LABEL_49;
  }
  v47 = (void *)v29;
  v36 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2675360, (__int64)&v47);
  ++*(_DWORD *)(a1 + 3556);
  v43 = v36;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v29 + 24LL))(v29, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v43 )
    sub_C9AF60(v43);
  if ( !v44 )
  {
LABEL_49:
    v42 = *(_BYTE *)(v29 + 400);
    *(_BYTE *)(v29 + 96) = 1;
    *(_BYTE *)(v29 + 464) = 1;
    *(_BYTE *)(v29 + 401) = v42;
    *(_BYTE *)(v29 + 337) = *(_BYTE *)(v29 + 336);
    *(_BYTE *)(v29 + 241) = *(_BYTE *)(v29 + 240);
    *(_BYTE *)(v29 + 113) = *(_BYTE *)(v29 + 112);
    *(_BYTE *)(v29 + 177) = *(_BYTE *)(v29 + 176);
    return v29;
  }
  if ( a6 )
  {
    v37 = *(_DWORD *)(a1 + 3552);
    *(_DWORD *)(a1 + 3552) = 1;
    sub_251C580(a1, v29);
    *(_DWORD *)(a1 + 3552) = v37;
  }
LABEL_39:
  if ( a4 )
    sub_250ED80(a1, v29, a4, a5);
  return v29;
}
