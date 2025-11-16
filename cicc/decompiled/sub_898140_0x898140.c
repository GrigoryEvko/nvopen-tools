// Function: sub_898140
// Address: 0x898140
//
__int64 __fastcall sub_898140(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        int a7,
        __m128i *a8,
        unsigned __int64 a9,
        unsigned int *a10,
        __int64 a11,
        _QWORD *a12)
{
  unsigned __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  char i; // dl
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  unsigned int v27; // r13d
  _QWORD *v28; // r14
  int v29; // edx
  __int64 v30; // rdx
  unsigned int *v31; // rsi
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 result; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v44; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  __int64 v49[7]; // [rsp+38h] [rbp-38h] BYREF

  *(_BYTE *)(a1 + 122) = (8 * ((a2 ^ 1) & 1)) | *(_BYTE *)(a1 + 122) & 0xF7;
  *(_QWORD *)(a1 + 184) = sub_5CC190(1);
  v15 = dword_4F0774C;
  if ( dword_4F0774C )
    *(_BYTE *)(a1 + 124) |= 0x40u;
  v16 = qword_4F061C8;
  v17 = (-(__int64)(dword_4F077BC == 0) & 0xFFFFFFFFFFC00000LL) + 4194355;
  if ( a2 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 81LL);
    v17 |= 0x200u;
    ++*(_BYTE *)(v16 + 63);
    ++*(_BYTE *)(v16 + 83);
    if ( !a6 && dword_4F077C4 == 2 )
      *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  }
  else
  {
    ++*(_BYTE *)(qword_4F061C8 + 17LL);
  }
  if ( a3 )
    v17 |= 4u;
  else
    *(_BYTE *)(a1 + 132) |= 0x80u;
  sub_672A20(v17, a1, (__int64)a12, v15, v14);
  v21 = *(_QWORD *)(a1 + 272);
  for ( i = *(_BYTE *)(v21 + 140); i == 12; i = *(_BYTE *)(v21 + 140) )
    v21 = *(_QWORD *)(v21 + 160);
  if ( !i )
  {
    if ( word_4F06418[0] != 1 )
    {
      if ( word_4F06418[0] == 27 || word_4F06418[0] == 34 )
        goto LABEL_18;
      if ( dword_4F077C4 == 2 )
      {
        if ( word_4F06418[0] == 33 || dword_4D04474 && word_4F06418[0] == 52 )
          goto LABEL_18;
        v20 = dword_4D0485C;
        if ( dword_4D0485C )
        {
          if ( word_4F06418[0] == 25 )
            goto LABEL_18;
        }
        if ( word_4F06418[0] == 156 )
          goto LABEL_18;
      }
LABEL_14:
      *a8 = _mm_loadu_si128(xmmword_4F06660);
      a8[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a8[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v23 = *(_QWORD *)dword_4F07508;
      a8[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
      a8[1].m128i_i8[1] |= 0x20u;
      a8->m128i_i64[1] = v23;
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_59;
    }
    if ( dword_4F077C4 == 2 )
    {
      if ( (word_4D04A10 & 0x200) != 0
        || (v17 = 0, (unsigned int)sub_7C0F00(0, 0, (__int64)&qword_4D04A00, v18, v19, v20)) )
      {
        if ( (unk_4D04A12 & 1) != 0 )
          goto LABEL_14;
      }
    }
  }
LABEL_18:
  if ( a2 )
  {
    v25 = a3;
    v26 = *(_QWORD *)(a1 + 8);
    v24 = (-(__int64)(a7 == 0) & 0xFFFFFFFFFFFFF000LL) + 4749;
    if ( !a3 )
      goto LABEL_39;
  }
  else
  {
    v24 = 525;
    if ( a4 && *(_BYTE *)(a4 + 80) == 21 )
      v24 = 524813;
    v25 = a3;
    v26 = *(_QWORD *)(a1 + 8);
    if ( !a3 )
      goto LABEL_39;
  }
  if ( (v26 & 0x400) != 0 )
  {
    v24 |= 0x20u;
    goto LABEL_25;
  }
LABEL_39:
  v29 = *(unsigned __int8 *)(a1 + 125);
  if ( (v29 & 4) != 0 && word_4F06418[0] == 27 )
  {
    v30 = v29 | 0x10u;
    *(_BYTE *)(a1 + 125) = v30;
    v31 = (unsigned int *)(v24 | 0x100000);
    if ( (v26 & 1) == 0 )
    {
      v32 = v24 | 0x110000;
      if ( (*(_BYTE *)(a1 + 120) & 0x7F) == 0 )
        v31 = (unsigned int *)v32;
    }
    sub_7B8B50(v25, v31, v30, (__int64)word_4F06418, a2, v20);
    sub_627530(a1, (unsigned __int64)v31, v49, (char *)a9, 0, a5, 0, 0, 0, 0, 0, 0, 1, (__int64)a12);
    *a8 = _mm_loadu_si128(xmmword_4F06660);
    a8[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a8[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v33 = *(_QWORD *)&dword_4F077C8;
    a8[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a8->m128i_i64[1] = v33;
    v34 = v49[0];
    if ( *(_BYTE *)(v49[0] + 140) == 7 )
      *(_QWORD *)(v49[0] + 160) = *(_QWORD *)(a1 + 272);
    *(_QWORD *)(a1 + 288) = v34;
    goto LABEL_51;
  }
LABEL_25:
  if ( (v26 & 1) == 0 && (*(_BYTE *)(a1 + 120) & 0x7F) == 0 )
    v24 |= 0x10000u;
  v44 = v24;
  v46 = a5;
  while ( 1 )
  {
    v27 = dword_4F06650[0];
    v28 = *(_QWORD **)(a1 + 432);
    sub_626F50(v44, a1, v46, (__int64)a8, a9, a12);
    if ( a6 )
    {
      a5 = v46;
      a8[1].m128i_i64[1] = 0;
      a8[1].m128i_i8[1] |= 0x20u;
      goto LABEL_50;
    }
    if ( *(char *)(a1 + 125) < 0
      || (*(_BYTE *)(a1 + 131) & 0x10) != 0
      || !*(_QWORD *)(a1 + 368)
      || (*(_BYTE *)(a1 + 133) & 8) != 0 )
    {
      break;
    }
    sub_897F40(v27, v28, a9, (__int64)a8);
  }
  a5 = v46;
LABEL_50:
  sub_88F5B0(a1, (__int64)a8);
LABEL_51:
  v35 = *(_QWORD **)(a1 + 352);
  if ( v35 )
  {
    sub_869FD0(v35, dword_4F04C64);
    *(_QWORD *)(a1 + 352) = 0;
  }
  *(_BYTE *)(a9 + 64) = *(_BYTE *)(a1 + 8) & 2 | *(_BYTE *)(a9 + 64) & 0xFD;
  v17 = *(_QWORD *)(a1 + 288);
  if ( (unsigned int)sub_8D2310(v17) )
  {
    v17 = *(_QWORD *)(a1 + 288);
    if ( *(_BYTE *)(v17 + 140) == 12 )
    {
      if ( (unsigned int)sub_8D4970(v17)
        || (v41 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 288) + 184LL), (unsigned __int8)v41 <= 0xCu)
        && (v42 = 6338, _bittest64(&v42, v41)) )
      {
        *(_BYTE *)(a9 + 64) |= 0x80u;
      }
    }
    if ( !a5 && (a8[1].m128i_i8[2] & 2) != 0 && *(_BYTE *)(a1 + 268) )
    {
      v17 = 8;
      sub_684AA0(8u, 0x50u, (_DWORD *)(a1 + 260));
      *(_BYTE *)(a1 + 269) = 0;
    }
    if ( *(_QWORD *)(a9 + 32) )
    {
      if ( !a2 && (*(_BYTE *)(*(_QWORD *)(a11 + 32) + 81LL) & 2) != 0 )
        *(_BYTE *)(a9 + 64) |= 4u;
      v17 = a9;
      sub_87E350(a9);
    }
    goto LABEL_59;
  }
  if ( (*(_BYTE *)(a1 + 10) & 8) != 0 )
  {
    v17 = *(_QWORD *)(a1 + 288);
    if ( (*(_BYTE *)(v17 + 140) & 0xFB) != 8 )
    {
LABEL_74:
      *(_QWORD *)(a1 + 288) = sub_73C570((const __m128i *)v17, 1);
      goto LABEL_75;
    }
    if ( (sub_8D4C10(v17, dword_4F077C4 != 2) & 1) == 0 )
    {
      v17 = *(_QWORD *)(a1 + 288);
      goto LABEL_74;
    }
  }
LABEL_75:
  if ( a8[2].m128i_i64[1] )
  {
    if ( (a8[1].m128i_i8[2] & 1) == 0 )
    {
      a8[2].m128i_i64[1] = 0;
      if ( (a8[1].m128i_i8[1] & 0x20) == 0 )
      {
        v17 = 891;
        sub_6851C0(0x37Bu, &a8->m128i_i32[2]);
        *a8 = _mm_loadu_si128(xmmword_4F06660);
        a8[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
        a8[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
        v39 = *(_QWORD *)dword_4F07508;
        a8[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
        a8[1].m128i_i8[1] |= 0x20u;
        a8->m128i_i64[1] = v39;
      }
    }
  }
LABEL_59:
  v36 = a2;
  v37 = qword_4F061C8;
  if ( a2 )
  {
    --*(_BYTE *)(qword_4F061C8 + 81LL);
    --*(_BYTE *)(v37 + 63);
    --*(_BYTE *)(v37 + 83);
    result = a6;
    if ( !a6 )
    {
      if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
        sub_87DD20(dword_4F04C40);
      result = (__int64)&dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        v40 = (int)dword_4F04C40;
        result = 776 * v40;
        *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) &= ~8u;
        if ( *(_QWORD *)(qword_4F04C68[0] + 776 * v40 + 456) )
          return sub_8845B0(v40);
      }
    }
  }
  else
  {
    --*(_BYTE *)(qword_4F061C8 + 17LL);
    if ( a10 )
    {
      v17 = a1 + 288;
      sub_88DF70((__int64 *)(a1 + 288), (__int64)a10, (a8[1].m128i_i8[1] & 0x20) != 0, 0, a11);
    }
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(v17, a10, v36, v18, v19, v20);
    return sub_7B8B50(v17, a10, v36, v18, v19, v20);
  }
  return result;
}
