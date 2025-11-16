// Function: sub_33CFB90
// Address: 0x33cfb90
//
char __fastcall sub_33CFB90(__int64 *a1, __int64 a2, int a3, int a4)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rcx
  char result; // al
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // r15
  __int64 v19; // rax
  int v20; // ecx
  unsigned int v21; // r14d
  __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-68h]
  __m128i v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+28h] [rbp-48h]

  v7 = *a1;
  if ( *a1 == a2 )
    goto LABEL_10;
  while ( 1 )
  {
    if ( !a4 )
      return 0;
    v8 = *(_DWORD *)(v7 + 24);
    if ( v8 == 2 )
      break;
    if ( v8 != 298 )
      return 0;
    v9 = *(_QWORD *)(v7 + 112);
    if ( (*(_BYTE *)(v9 + 37) & 0xFu) > 1 || (*(_BYTE *)(v9 + 32) & 4) != 0 )
      return 0;
    a1 = *(__int64 **)(v7 + 40);
    --a4;
    v7 = *a1;
    if ( *a1 == a2 )
    {
LABEL_10:
      if ( *((_DWORD *)a1 + 2) == a3 )
        return 1;
    }
  }
  v11 = *(_QWORD *)(v7 + 40);
  v12 = 40LL * *(unsigned int *)(v7 + 64);
  v23 = v11 + v12;
  v13 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v12 >> 3)) >> 2;
  if ( !v13 )
  {
    v15 = *(_QWORD *)(v7 + 40);
LABEL_19:
    v16 = v23 - v15;
    if ( v23 - v15 != 80 )
    {
      if ( v16 != 120 )
      {
        if ( v16 != 40 )
          goto LABEL_22;
LABEL_65:
        if ( *(_QWORD *)v15 != a2 || *(_DWORD *)(v15 + 8) != a3 )
          goto LABEL_22;
        goto LABEL_67;
      }
      if ( *(_QWORD *)v15 == a2 && *(_DWORD *)(v15 + 8) == a3 )
      {
LABEL_67:
        if ( v23 == v15 )
          goto LABEL_22;
        v19 = *(_QWORD *)(a2 + 56);
        if ( !v19 )
          goto LABEL_22;
        goto LABEL_33;
      }
      v15 += 40;
    }
    if ( *(_QWORD *)v15 != a2 || *(_DWORD *)(v15 + 8) != a3 )
    {
      v15 += 40;
      goto LABEL_65;
    }
    goto LABEL_67;
  }
  v14 = *(_QWORD *)(v7 + 40);
  v15 = v11 + 160 * v13;
  while ( *(_QWORD *)v14 != a2 || *(_DWORD *)(v14 + 8) != a3 )
  {
    if ( *(_QWORD *)(v14 + 40) == a2 && *(_DWORD *)(v14 + 48) == a3 )
    {
      v14 += 40;
      break;
    }
    if ( *(_QWORD *)(v14 + 80) == a2 && *(_DWORD *)(v14 + 88) == a3 )
    {
      v14 += 80;
      break;
    }
    if ( *(_QWORD *)(v14 + 120) == a2 && *(_DWORD *)(v14 + 128) == a3 )
    {
      v14 += 120;
      break;
    }
    v14 += 160;
    if ( v14 == v15 )
      goto LABEL_19;
  }
  if ( v23 != v14 )
  {
    v19 = *(_QWORD *)(a2 + 56);
    if ( v19 )
    {
LABEL_33:
      v20 = 1;
      do
      {
        if ( *(_DWORD *)(v19 + 8) == a3 )
        {
          if ( !v20 )
            goto LABEL_22;
          v19 = *(_QWORD *)(v19 + 32);
          if ( !v19 )
            return 1;
          if ( a3 == *(_DWORD *)(v19 + 8) )
            goto LABEL_22;
          v20 = 0;
        }
        v19 = *(_QWORD *)(v19 + 32);
      }
      while ( v19 );
      if ( v20 != 1 )
        return 1;
LABEL_22:
      v25 = a2;
      LODWORD(v26) = a3;
      if ( v13 )
        goto LABEL_23;
LABEL_49:
      v21 = a4 - 1;
      v22 = v23 - v11;
      if ( v23 - v11 != 80 )
      {
        if ( v22 != 120 )
        {
          result = 1;
          v21 = a4 - 1;
          if ( v22 != 40 )
            return result;
LABEL_52:
          v24 = _mm_loadu_si128((const __m128i *)v11);
          result = sub_33CFB90(&v24, v25, v26, v21);
          if ( result )
            return result;
          return v23 == v11;
        }
        v21 = a4 - 1;
        v24 = _mm_loadu_si128((const __m128i *)v11);
        if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, (unsigned int)(a4 - 1)) )
          return v23 == v11;
        v11 += 40;
      }
      v24 = _mm_loadu_si128((const __m128i *)v11);
      if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, v21) )
        return v23 == v11;
      v11 += 40;
      goto LABEL_52;
    }
  }
  v25 = a2;
  LODWORD(v26) = a3;
LABEL_23:
  v17 = a4 - 1;
  v18 = v11 + 160 * v13;
  while ( 1 )
  {
    v24 = _mm_loadu_si128((const __m128i *)v11);
    if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, v17) )
      return v11 == v23;
    v24 = _mm_loadu_si128((const __m128i *)(v11 + 40));
    if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, v17) )
      return v23 == v11 + 40;
    v24 = _mm_loadu_si128((const __m128i *)(v11 + 80));
    if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, v17) )
      return v23 == v11 + 80;
    v24 = _mm_loadu_si128((const __m128i *)(v11 + 120));
    if ( !(unsigned __int8)sub_33CFB90(&v24, v25, v26, v17) )
      return v23 == v11 + 120;
    v11 += 160;
    if ( v18 == v11 )
      goto LABEL_49;
  }
}
