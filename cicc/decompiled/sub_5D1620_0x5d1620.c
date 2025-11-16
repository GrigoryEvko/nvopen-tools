// Function: sub_5D1620
// Address: 0x5d1620
//
__int64 __fastcall sub_5D1620(__int64 a1, __int64 a2)
{
  int v2; // ecx
  const __m128i *v3; // r13
  __int64 v4; // r12
  __int64 v5; // rbx
  char v6; // dl
  __int64 v7; // rax
  int v8; // eax
  bool v9; // zf
  __int64 result; // rax
  __int64 *v11; // rax
  __int64 *v12; // rdx
  char v13; // si
  char v14; // si
  char v15; // si
  __int64 v16; // rax
  __m128i v17; // xmm4
  unsigned int v18; // esi
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = 0;
  v3 = 0;
  v4 = a2;
  v5 = a1;
  v6 = *(_BYTE *)(a2 + 140);
  for ( v19[0] = a1; v6 == 12; v6 = *(_BYTE *)(v4 + 140) )
  {
    if ( (*(_BYTE *)(v4 + 143) & 2) != 0 && !v2 )
    {
      v7 = sub_736C60(50, *(_QWORD *)(v4 + 104));
      v2 = 1;
      v3 = (const __m128i *)v7;
    }
    v4 = *(_QWORD *)(v4 + 160);
  }
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    v8 = 0;
    do
    {
      v9 = (*(_BYTE *)(v5 + 143) & 2) == 0;
      v5 = *(_QWORD *)(v5 + 160);
      if ( !v9 )
        v8 = 1;
    }
    while ( *(_BYTE *)(v5 + 140) == 12 );
    v2 &= v8 ^ 1;
  }
  if ( v6 == 7 && v4 != v5 )
  {
    v11 = *(__int64 **)(v4 + 168);
    v12 = *(__int64 **)(v5 + 168);
    if ( *(char *)(v4 + 142) < 0 )
    {
      v18 = *(_DWORD *)(v4 + 136);
      if ( v18 > *(_DWORD *)(v5 + 136) )
      {
        *(_BYTE *)(v5 + 142) |= 0x80u;
        *(_DWORD *)(v5 + 136) = v18;
      }
    }
    v13 = *((_BYTE *)v11 + 25);
    if ( v13 && *((_BYTE *)v12 + 25) != 3 )
    {
      *((_BYTE *)v12 + 25) = v13;
      *((_BYTE *)v12 + 20) = *((_BYTE *)v11 + 20) & 0x20 | *((_BYTE *)v12 + 20) & 0xDF;
    }
    v14 = *((_BYTE *)v11 + 20);
    if ( (v14 & 8) != 0 )
    {
      *((_BYTE *)v12 + 20) |= 8u;
      v14 = *((_BYTE *)v11 + 20);
    }
    if ( (v14 & 4) != 0 )
      *((_BYTE *)v12 + 20) |= 4u;
    v15 = *((_BYTE *)v11 + 24);
    if ( v15 )
    {
      *((_BYTE *)v12 + 24) = v15;
      *((_DWORD *)v12 + 7) = *((_DWORD *)v11 + 7);
      *((_DWORD *)v12 + 8) = *((_DWORD *)v11 + 8);
    }
    if ( (v11[2] & 2) != 0 && (v12[2] & 2) != 0 )
    {
      while ( 1 )
      {
        v11 = (__int64 *)*v11;
        v12 = (__int64 *)*v12;
        if ( !v11 )
          break;
        while ( (*((_BYTE *)v11 + 34) & 8) != 0 )
        {
          *((_BYTE *)v12 + 34) |= 8u;
          v11 = (__int64 *)*v11;
          v12 = (__int64 *)*v12;
          if ( !v11 )
            goto LABEL_34;
        }
      }
    }
LABEL_34:
    if ( (*(_BYTE *)(v4 + 143) & 2) != 0 )
      *(_BYTE *)(v5 + 143) |= 2u;
    v19[0] = v5;
    result = v5;
  }
  else
  {
    result = v19[0];
  }
  if ( v2 )
  {
    if ( (*(_BYTE *)(result + 143) & 2) == 0 )
    {
      v16 = sub_727670();
      *(__m128i *)v16 = _mm_loadu_si128(v3);
      *(__m128i *)(v16 + 16) = _mm_loadu_si128(v3 + 1);
      *(__m128i *)(v16 + 32) = _mm_loadu_si128(v3 + 2);
      *(__m128i *)(v16 + 48) = _mm_loadu_si128(v3 + 3);
      v17 = _mm_loadu_si128(v3 + 4);
      *(_QWORD *)v16 = 0;
      *(__m128i *)(v16 + 64) = v17;
      sub_5CF030(v19, (_QWORD *)v16, 0);
      return v19[0];
    }
  }
  return result;
}
