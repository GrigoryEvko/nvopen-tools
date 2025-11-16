// Function: sub_102C4C0
// Address: 0x102c4c0
//
__int64 __fastcall sub_102C4C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  unsigned int v20; // eax
  _QWORD **v21; // r12
  _QWORD **i; // rbx
  __int64 v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // r14
  __int64 v26; // rdi
  unsigned int v27; // eax
  _QWORD *v28; // rbx
  _QWORD *v29; // r12
  __int64 v30; // rdi
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  int v35; // ecx
  _QWORD *v36; // rax
  __m128i v38; // xmm5
  __m128i v39; // xmm6
  __m128i v40; // xmm7
  __m128i v41; // xmm0
  __m128i v42; // xmm1
  __int64 v43; // [rsp+8h] [rbp-F8h]
  __int64 v44; // [rsp+10h] [rbp-F0h]
  __int64 v45; // [rsp+18h] [rbp-E8h]
  __m128i v46; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v47; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v48; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v49; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v50; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE v51[8]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v52; // [rsp+78h] [rbp-88h]
  unsigned int v53; // [rsp+88h] [rbp-78h]
  __int64 v54; // [rsp+98h] [rbp-68h]
  unsigned int v55; // [rsp+A8h] [rbp-58h]
  __int64 v56; // [rsp+B8h] [rbp-48h]
  unsigned int v57; // [rsp+C8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F86530 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_55;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F86530);
  v7 = *(__int64 **)(a1 + 8);
  v43 = *(_QWORD *)(v6 + 176);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_52:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8662C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_52;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8662C);
  v11 = sub_CFFAC0(v10, a2);
  v12 = *(__int64 **)(a1 + 8);
  v45 = v11;
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_53:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F6D3F0 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_53;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v51);
  sub_983BD0((__int64)&v46, v15 + 176, a2);
  v44 = v15 + 408;
  if ( *(_BYTE *)(v15 + 488) )
  {
    v16 = _mm_loadu_si128(&v47);
    v17 = _mm_loadu_si128(&v48);
    v18 = _mm_loadu_si128(&v49);
    v19 = _mm_loadu_si128(&v50);
    *(__m128i *)(v15 + 408) = _mm_loadu_si128(&v46);
    *(__m128i *)(v15 + 424) = v16;
    *(__m128i *)(v15 + 440) = v17;
    *(__m128i *)(v15 + 456) = v18;
    *(__m128i *)(v15 + 472) = v19;
  }
  else
  {
    v38 = _mm_loadu_si128(&v46);
    v39 = _mm_loadu_si128(&v47);
    *(_BYTE *)(v15 + 488) = 1;
    v40 = _mm_loadu_si128(&v48);
    v41 = _mm_loadu_si128(&v49);
    v42 = _mm_loadu_si128(&v50);
    *(__m128i *)(v15 + 408) = v38;
    *(__m128i *)(v15 + 424) = v39;
    *(__m128i *)(v15 + 440) = v40;
    *(__m128i *)(v15 + 456) = v41;
    *(__m128i *)(v15 + 472) = v42;
  }
  sub_C7D6A0(v56, 24LL * v57, 8);
  v20 = v55;
  if ( v55 )
  {
    v21 = (_QWORD **)(v54 + 32LL * v55);
    for ( i = (_QWORD **)(v54 + 8); ; i += 4 )
    {
      v23 = (__int64)*(i - 1);
      if ( v23 != -8192 && v23 != -4096 )
      {
        v24 = *i;
        while ( v24 != i )
        {
          v25 = v24;
          v24 = (_QWORD *)*v24;
          v26 = v25[3];
          if ( v26 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
          j_j___libc_free_0(v25, 32);
        }
      }
      if ( v21 == i + 3 )
        break;
    }
    v20 = v55;
  }
  sub_C7D6A0(v54, 32LL * v20, 8);
  v27 = v53;
  if ( v53 )
  {
    v28 = v52;
    v29 = &v52[2 * v53];
    do
    {
      if ( *v28 != -4096 && *v28 != -8192 )
      {
        v30 = v28[1];
        if ( v30 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
      }
      v28 += 2;
    }
    while ( v29 != v28 );
    v27 = v53;
  }
  sub_C7D6A0((__int64)v52, 16LL * v27, 8);
  v31 = *(__int64 **)(a1 + 8);
  v32 = *v31;
  v33 = v31[1];
  if ( v32 == v33 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F8144C )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_54;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_4F8144C)
      + 176;
  if ( *(_BYTE *)(a1 + 1192) )
  {
    *(_BYTE *)(a1 + 1192) = 0;
    sub_102BD40(a1 + 176);
  }
  *(_QWORD *)(a1 + 176) = 0;
  v35 = qword_4F8F288;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_DWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_DWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 456) = v34;
  *(_QWORD *)(a1 + 432) = v43;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 440) = v45;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 448) = v44;
  *(_QWORD *)(a1 + 512) = a1 + 528;
  *(_QWORD *)(a1 + 520) = 0x400000000LL;
  *(_QWORD *)(a1 + 560) = a1 + 576;
  *(_QWORD *)(a1 + 480) = 0;
  *(_DWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 592) = &unk_49DDC10;
  v36 = (_QWORD *)(a1 + 696);
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 1;
  *(_QWORD *)(a1 + 600) = v34;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_DWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_DWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 668) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 1;
  do
  {
    if ( v36 )
      *v36 = -4096;
    v36 += 11;
  }
  while ( v36 != (_QWORD *)(a1 + 1048) );
  *(_QWORD *)(a1 + 1048) = 0;
  *(_QWORD *)(a1 + 1056) = a1 + 1080;
  *(_QWORD *)(a1 + 1064) = 8;
  *(_DWORD *)(a1 + 1072) = 0;
  *(_BYTE *)(a1 + 1076) = 1;
  *(_BYTE *)(a1 + 1144) = 0;
  *(_DWORD *)(a1 + 1152) = v35;
  *(_QWORD *)(a1 + 1160) = 0;
  *(_QWORD *)(a1 + 1168) = 0;
  *(_QWORD *)(a1 + 1176) = 0;
  *(_DWORD *)(a1 + 1184) = 0;
  *(_BYTE *)(a1 + 1192) = 1;
  return 0;
}
