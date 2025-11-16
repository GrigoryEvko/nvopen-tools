// Function: sub_C2CAB0
// Address: 0xc2cab0
//
__int64 __fastcall sub_C2CAB0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r12
  __int128 v13; // rax
  void *v14; // rax
  void *v15; // rdx
  unsigned int v16; // r13d
  int v17; // esi
  __m128i *v18; // r10
  int v19; // eax
  __int64 v21; // r14
  int v22; // eax
  unsigned int v23; // ecx
  void *v24; // rdx
  int v25; // eax
  const void *v26; // r13
  int v27; // r11d
  unsigned int k; // r9d
  __int64 v29; // r12
  const void *v30; // rsi
  unsigned int v31; // r9d
  int v32; // eax
  unsigned int v33; // ecx
  unsigned int v34; // eax
  _QWORD *v35; // rdi
  int v36; // ebx
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *j; // rdx
  int v42; // eax
  _QWORD *v43; // rax
  void *v44; // [rsp+0h] [rbp-90h]
  __int64 v45; // [rsp+10h] [rbp-80h]
  int v46; // [rsp+1Ch] [rbp-74h]
  unsigned int v47; // [rsp+20h] [rbp-70h]
  unsigned int v48; // [rsp+24h] [rbp-6Ch]
  __m128i *v49; // [rsp+28h] [rbp-68h]
  __int64 v50; // [rsp+30h] [rbp-60h]
  __m128i *v51; // [rsp+48h] [rbp-48h] BYREF
  void *s1[2]; // [rsp+50h] [rbp-40h] BYREF

  v45 = a1 + 488;
  v2 = *(_DWORD *)(a1 + 504);
  ++*(_QWORD *)(a1 + 488);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 508) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 512);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 496), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 496) = 0;
      *(_QWORD *)(a1 + 504) = 0;
      *(_DWORD *)(a1 + 512) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v33 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 512);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v33 = 64;
  if ( (unsigned int)v3 <= v33 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 496);
    for ( i = &v4[2 * v3]; i != v4; *(v4 - 1) = 0 )
    {
      *v4 = -1;
      v4 += 2;
    }
    *(_QWORD *)(a1 + 504) = 0;
    goto LABEL_7;
  }
  v34 = v2 - 1;
  if ( !v34 )
  {
    v35 = *(_QWORD **)(a1 + 496);
    v36 = 64;
LABEL_41:
    sub_C7D6A0(v35, 16LL * (unsigned int)v3, 8);
    v37 = ((((((((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
             | (4 * v36 / 3u + 1)
             | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 4)
           | (((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
           | (4 * v36 / 3u + 1)
           | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
           | (4 * v36 / 3u + 1)
           | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 4)
         | (((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
         | (4 * v36 / 3u + 1)
         | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 16;
    v38 = (v37
         | (((((((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
             | (4 * v36 / 3u + 1)
             | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 4)
           | (((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
           | (4 * v36 / 3u + 1)
           | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
           | (4 * v36 / 3u + 1)
           | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 4)
         | (((4 * v36 / 3u + 1) | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)) >> 2)
         | (4 * v36 / 3u + 1)
         | ((unsigned __int64)(4 * v36 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 512) = v38;
    v39 = (_QWORD *)sub_C7D670(16 * v38, 8);
    v40 = *(unsigned int *)(a1 + 512);
    *(_QWORD *)(a1 + 504) = 0;
    *(_QWORD *)(a1 + 496) = v39;
    for ( j = &v39[2 * v40]; j != v39; v39 += 2 )
    {
      if ( v39 )
      {
        *v39 = -1;
        v39[1] = 0;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v34, v34);
  v35 = *(_QWORD **)(a1 + 496);
  v36 = 1 << (33 - (v34 ^ 0x1F));
  if ( v36 < 64 )
    v36 = 64;
  if ( v36 != (_DWORD)v3 )
    goto LABEL_41;
  *(_QWORD *)(a1 + 504) = 0;
  v43 = &v35[2 * (unsigned int)v36];
  do
  {
    if ( v35 )
    {
      *v35 = -1;
      v35[1] = 0;
    }
    v35 += 2;
  }
  while ( v43 != v35 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 192);
  v7 = *(_QWORD *)(v6 + 32);
  v50 = v6 + 24;
  if ( v7 == v6 + 24 )
    return 1;
  do
  {
    while ( 2 )
    {
      v8 = v7 - 56;
      if ( !v7 )
        v8 = 0;
      s1[0] = (void *)sub_B2D7E0(v8, "sample-profile-suffix-elision-policy", 0x24u);
      v9 = sub_A72240((__int64 *)s1);
      v11 = v10;
      v12 = v9;
      *(_QWORD *)&v13 = sub_BD5D20(v8);
      v14 = (void *)sub_C16140(v13, v12, v11);
      v16 = *(_DWORD *)(a1 + 512);
      s1[0] = v14;
      s1[1] = v15;
      if ( !v16 )
      {
        ++*(_QWORD *)(a1 + 488);
        v51 = 0;
LABEL_12:
        v17 = 2 * v16;
        goto LABEL_13;
      }
      v21 = *(_QWORD *)(a1 + 496);
      v22 = sub_C94890(v14, v15);
      v23 = v16 - 1;
      v24 = s1[1];
      v18 = 0;
      v25 = (v16 - 1) & v22;
      v26 = s1[0];
      v27 = 1;
      for ( k = v25; ; k = v23 & v31 )
      {
        v29 = v21 + 16LL * k;
        v30 = *(const void **)v29;
        if ( *(_QWORD *)v29 == -1 )
          break;
        if ( v30 == (const void *)-2LL )
        {
          if ( v26 == (const void *)-2LL )
            goto LABEL_24;
        }
        else
        {
          if ( *(void **)(v29 + 8) != v24 )
            goto LABEL_22;
          v46 = v27;
          v47 = k;
          v48 = v23;
          v49 = v18;
          if ( !v24 )
            goto LABEL_24;
          v44 = v24;
          v32 = memcmp(v26, v30, (size_t)v24);
          v24 = v44;
          v18 = v49;
          v23 = v48;
          k = v47;
          v27 = v46;
          if ( !v32 )
            goto LABEL_24;
        }
        if ( !v18 && v30 == (const void *)-2LL )
          v18 = (__m128i *)v29;
LABEL_22:
        v31 = v27 + k;
        ++v27;
      }
      if ( v26 == (const void *)-1LL )
      {
LABEL_24:
        v7 = *(_QWORD *)(v7 + 8);
        if ( v50 != v7 )
          continue;
        return 1;
      }
      break;
    }
    v42 = *(_DWORD *)(a1 + 504);
    v16 = *(_DWORD *)(a1 + 512);
    if ( !v18 )
      v18 = (__m128i *)(v21 + 16LL * k);
    ++*(_QWORD *)(a1 + 488);
    v19 = v42 + 1;
    v51 = v18;
    if ( 4 * v19 >= 3 * v16 )
      goto LABEL_12;
    if ( v16 - (v19 + *(_DWORD *)(a1 + 508)) > v16 >> 3 )
      goto LABEL_14;
    v17 = v16;
LABEL_13:
    sub_BA8070(v45, v17);
    sub_B9B010(v45, s1, &v51);
    v18 = v51;
    v19 = *(_DWORD *)(a1 + 504) + 1;
LABEL_14:
    *(_DWORD *)(a1 + 504) = v19;
    if ( v18->m128i_i64[0] != -1 )
      --*(_DWORD *)(a1 + 508);
    *v18 = _mm_loadu_si128((const __m128i *)s1);
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v50 != v7 );
  return 1;
}
