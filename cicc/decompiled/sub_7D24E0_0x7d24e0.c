// Function: sub_7D24E0
// Address: 0x7d24e0
//
const __m128i *__fastcall sub_7D24E0(__int64 a1, _QWORD *a2, int a3, int a4)
{
  _QWORD *v4; // r14
  __int64 v5; // r15
  char v6; // al
  __int64 i; // rdi
  char v8; // al
  __int64 v9; // r13
  char v10; // al
  __int64 v11; // r12
  const __m128i *v12; // r12
  __int64 v13; // rcx
  __int64 j; // rax
  unsigned __int64 *v15; // rdi
  __int64 v16; // r14
  __int64 v17; // r15
  unsigned __int64 v18; // rbx
  __int64 v20; // rax
  __int64 v21; // r13
  __m128i v22; // xmm6
  __int64 v23; // rax
  char v24; // al
  __int64 v25; // r12
  char v26; // dl
  char v27; // al
  __int64 v28; // rbx
  __int64 v29; // rax
  int v33; // [rsp+24h] [rbp-4Ch] BYREF
  unsigned __int64 *v34; // [rsp+28h] [rbp-48h] BYREF
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v36[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a1 + 56);
  v34 = 0;
  if ( !unk_4F0774C )
  {
    if ( !a2 )
      return 0;
    goto LABEL_9;
  }
  v6 = *(_BYTE *)(v5 + 140);
  for ( i = v5; v6 == 12; v6 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( v6 == 6 )
  {
    do
    {
      do
      {
        i = *(_QWORD *)(i + 160);
        v8 = *(_BYTE *)(i + 140);
      }
      while ( v8 == 12 );
    }
    while ( v8 == 6 );
  }
  if ( (unsigned int)sub_8D3EA0(i) )
    return 0;
  if ( a2 )
  {
    do
    {
LABEL_9:
      v36[0] = 0;
      v9 = v4[1];
      v10 = *(_BYTE *)(v9 + 80);
      v11 = v9;
      if ( v10 == 16 )
      {
        v11 = **(_QWORD **)(v9 + 88);
        v10 = *(_BYTE *)(v11 + 80);
      }
      if ( v10 == 24 )
      {
        v11 = *(_QWORD *)(v11 + 88);
        v10 = *(_BYTE *)(v11 + 80);
      }
      switch ( v10 )
      {
        case 4:
        case 5:
          v13 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 80LL);
          break;
        case 6:
          v13 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 32LL);
          break;
        case 9:
        case 10:
          v13 = *(_QWORD *)(*(_QWORD *)(v11 + 96) + 56LL);
          break;
        case 19:
        case 20:
        case 21:
        case 22:
          v13 = *(_QWORD *)(v11 + 88);
          break;
        default:
          v13 = 0;
          break;
      }
      for ( j = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 88) + 176LL) + 152LL);
            *(_BYTE *)(j + 140) == 12;
            j = *(_QWORD *)(j + 160) )
      {
        ;
      }
      if ( !a3 || (*(_BYTE *)(*(_QWORD *)(j + 168) + 18LL) & 0x7F) == a4 )
      {
        if ( (unsigned int)sub_8B3500(v5, *(_QWORD *)(j + 160), v36, **(_QWORD **)(v13 + 328), 0)
          && sub_8B2240(v36, v11, 0, 0x20000, 0) )
        {
          sub_8B5FF0(&v34, v9, v36[0]);
        }
        else if ( v36[0] )
        {
          sub_725130(v36[0]);
        }
      }
      v4 = (_QWORD *)*v4;
    }
    while ( v4 );
  }
  v15 = v34;
  if ( !v34 )
    return 0;
  v33 = 0;
  v16 = 0;
  v35 = 0;
  v36[0] = 0;
  v17 = a1 + 8;
  if ( a3 )
  {
    sub_893120(v34, 0, &v35, v36, &v33, 0);
    v12 = (const __m128i *)sub_8B74F0(v35, v36, *(_BYTE *)(a1 + 18) & 1, v17);
    if ( v33 || (*(_BYTE *)(v35 + 82) & 4) != 0 )
    {
      v28 = sub_87EBB0(10, v12->m128i_i64[0]);
      sub_877E20(v28, 0, v12[4].m128i_i64[0]);
      *(_BYTE *)(v28 + 82) |= 4u;
      *(_QWORD *)(v28 + 88) = v12[5].m128i_i64[1];
      v29 = v12[6].m128i_i64[0];
      v12 = (const __m128i *)v28;
      *(_QWORD *)(v28 + 96) = v29;
    }
  }
  else
  {
    while ( 1 )
    {
      v18 = *v15;
      *v15 = 0;
      v12 = (const __m128i *)sub_8B74F0(v15[1], v15 + 2, *(_BYTE *)(a1 + 18) & 1, v17);
      if ( !(v16 | v18) )
        break;
      v20 = sub_87EBB0(10, v12->m128i_i64[0]);
      v21 = v20;
      *(__m128i *)v20 = _mm_loadu_si128(v12);
      *(__m128i *)(v20 + 16) = _mm_loadu_si128(v12 + 1);
      *(__m128i *)(v20 + 32) = _mm_loadu_si128(v12 + 2);
      *(__m128i *)(v20 + 48) = _mm_loadu_si128(v12 + 3);
      *(__m128i *)(v20 + 64) = _mm_loadu_si128(v12 + 4);
      *(__m128i *)(v20 + 80) = _mm_loadu_si128(v12 + 5);
      v22 = _mm_loadu_si128(v12 + 6);
      *(_BYTE *)(v20 + 83) |= 1u;
      *(_QWORD *)(v20 + 8) = 0;
      *(__m128i *)(v20 + 96) = v22;
      if ( v16 )
      {
        if ( *(_BYTE *)(v16 + 80) == 17 )
        {
          v23 = *(_QWORD *)(v16 + 88);
        }
        else
        {
          v25 = sub_87EBB0(17, *(_QWORD *)v20);
          *(_DWORD *)(v25 + 40) = *(_DWORD *)(v21 + 40);
          *(_DWORD *)(v25 + 44) = *(_DWORD *)(v21 + 44);
          sub_877E20(v25, 0, *(_QWORD *)(v16 + 64));
          v26 = *(_BYTE *)(v21 + 84);
          v27 = *(_BYTE *)(v25 + 84);
          *(_QWORD *)(v25 + 88) = v16;
          *(_BYTE *)(v25 + 84) = v26 & 4 | v27 & 0xFB;
          v23 = v16;
          v16 = v25;
        }
        *(_QWORD *)(v21 + 8) = v23;
        v24 = *(_BYTE *)(v16 + 84);
        *(_QWORD *)(v16 + 88) = v21;
        *(_BYTE *)(v16 + 84) = (v24 | *(_BYTE *)(v21 + 84)) & 4 | v24 & 0xFB;
        if ( !v18 )
          return (const __m128i *)v16;
      }
      else
      {
        v16 = v20;
      }
      v15 = (unsigned __int64 *)v18;
    }
  }
  return v12;
}
