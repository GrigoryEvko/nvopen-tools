// Function: sub_3267350
// Address: 0x3267350
//
__int64 __fastcall sub_3267350(__int64 a1, __int64 a2)
{
  int v3; // r13d
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  bool v7; // sf
  __m128i v8; // xmm0
  unsigned __int64 v9; // rax
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // r14
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  __int64 v16; // rdx
  char v17; // cl
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rcx
  unsigned int v23; // esi
  _QWORD *v24; // rdx
  __int64 v25; // rdx
  __int64 *v26; // rsi
  unsigned int v27; // edx
  unsigned __int16 v28; // [rsp+0h] [rbp-50h] BYREF
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 104);
  if ( v3 == 298 )
  {
    v6 = 80;
  }
  else
  {
    v6 = 120;
    if ( v3 != 299 )
    {
      v7 = v5 < 0;
      v8 = _mm_loadu_si128((const __m128i *)(v4 + 40));
      *(_WORD *)a1 = 0;
      if ( v5 < 0 )
        v5 = 0;
      *(__m128i *)(a1 + 8) = v8;
      *(_QWORD *)(a1 + 24) = v5;
      if ( v7 )
      {
        *(_QWORD *)(a1 + 32) = -1;
      }
      else
      {
        v9 = *(_QWORD *)(a2 + 96);
        if ( v9 > 0x3FFFFFFFFFFFFFFBLL )
          v9 = 0xBFFFFFFFFFFFFFFELL;
        *(_QWORD *)(a1 + 32) = v9;
      }
      *(_QWORD *)(a1 + 40) = 0;
      return a1;
    }
  }
  v11 = *(_QWORD *)(v4 + v6);
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 == 35 || (v13 = 0, v12 == 11) )
  {
    v21 = (*(_WORD *)(a2 + 32) >> 7) & 7;
    if ( v21 == 1 )
    {
      v25 = *(_QWORD *)(v11 + 96);
      v26 = *(__int64 **)(v25 + 24);
      v27 = *(_DWORD *)(v25 + 32);
      if ( v27 > 0x40 )
      {
        v13 = *v26;
      }
      else
      {
        v13 = 0;
        if ( v27 )
          v13 = (__int64)((_QWORD)v26 << (64 - (unsigned __int8)v27)) >> (64 - (unsigned __int8)v27);
      }
    }
    else
    {
      v13 = 0;
      if ( v21 == 2 )
      {
        v22 = *(_QWORD *)(v11 + 96);
        v23 = *(_DWORD *)(v22 + 32);
        v24 = *(_QWORD **)(v22 + 24);
        if ( v23 > 0x40 )
        {
          v13 = -*v24;
        }
        else if ( v23 )
        {
          v13 = -((__int64)((_QWORD)v24 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23));
        }
      }
    }
  }
  v14 = *(_WORD *)(a2 + 96);
  v29 = v5;
  v28 = v14;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      BUG();
    v15 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
    v17 = byte_444C4A0[16 * v14 - 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v28);
    v30 = v15;
    v31 = v16;
    v17 = v16;
  }
  *(_BYTE *)a1 = (*(_BYTE *)(a2 + 32) & 8) != 0;
  v18 = *(_QWORD *)(a2 + 112);
  *(_BYTE *)(a1 + 1) = (*(_BYTE *)(v18 + 37) & 0xF) != 0;
  if ( (v3 & 0xFFFFFFBF) == 0x12B )
    *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(v4 + 80));
  else
    *(__m128i *)(a1 + 8) = _mm_loadu_si128((const __m128i *)(v4 + 40));
  *(_QWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 40) = v18;
  v19 = (unsigned __int64)(v15 + 7) >> 3;
  v20 = v19 | 0x4000000000000000LL;
  if ( !v17 )
    v20 = v19;
  *(_QWORD *)(a1 + 32) = v20;
  return a1;
}
