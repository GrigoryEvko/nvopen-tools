// Function: sub_24D5590
// Address: 0x24d5590
//
__int64 *__fastcall sub_24D5590(__int64 a1, unsigned int a2)
{
  __int64 v4; // r12
  char v5; // si
  unsigned __int64 v6; // rdx
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  _BYTE *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  _BYTE v18[352]; // [rsp+0h] [rbp-160h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v12 = a1 + 16;
    v13 = a1 + 336;
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    a2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v12 = a1 + 16;
      v13 = a1 + 336;
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 40LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v8 = 2560;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 40LL * v7;
        sub_24D53B0(a1, v4, (const __m128i *)(v4 + v10));
        return (__int64 *)sub_C7D6A0(v4, v10, 8);
      }
      v12 = a1 + 16;
      v13 = a1 + 336;
      a2 = 64;
    }
  }
  v14 = v18;
  do
  {
    v15 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 != -4096 && v15 != -8192 )
    {
      if ( v14 )
        *(_QWORD *)v14 = v15;
      *((_QWORD *)v14 + 1) = v14 + 24;
      v16 = *(_QWORD *)(v12 + 8);
      if ( v16 == v12 + 24 )
      {
        *(__m128i *)(v14 + 24) = _mm_loadu_si128((const __m128i *)(v12 + 24));
      }
      else
      {
        *((_QWORD *)v14 + 1) = v16;
        *((_QWORD *)v14 + 3) = *(_QWORD *)(v12 + 24);
      }
      v14 += 40;
      *((_QWORD *)v14 - 3) = *(_QWORD *)(v12 + 16);
    }
    v12 += 40;
  }
  while ( v12 != v13 );
  if ( a2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v17 = sub_C7D670(40LL * a2, 8);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v17;
  }
  return sub_24D53B0(a1, (__int64)v18, (const __m128i *)v14);
}
