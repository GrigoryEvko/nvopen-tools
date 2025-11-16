// Function: sub_D5BC90
// Address: 0xd5bc90
//
__m128i *__fastcall sub_D5BC90(__m128i *a1, __int64 a2, unsigned __int8 a3, __int64 *a4)
{
  const char *v7; // rbx
  int i; // edx
  __int64 v9; // r13
  __int64 v10; // rax
  int v11; // edx
  int v12; // r14d
  __m128i v13; // xmm0
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // r14
  unsigned int v17[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( !a4
    || !sub_981210(*a4, a2, v17)
    || (a4[((unsigned __int64)v17[0] >> 6) + 1] & (1LL << SLOBYTE(v17[0]))) != 0
    || (((int)*(unsigned __int8 *)(*a4 + (v17[0] >> 2)) >> (2 * (v17[0] & 3))) & 3) == 0 )
  {
    goto LABEL_3;
  }
  v7 = (const char *)&unk_3F72040;
  for ( i = 50; v17[0] != i; i = *(_DWORD *)v7 )
  {
    if ( v17[0] == *((_DWORD *)v7 + 7) )
    {
      v7 += 28;
      break;
    }
    if ( v17[0] == *((_DWORD *)v7 + 14) )
    {
      v7 += 56;
      break;
    }
    if ( v17[0] == *((_DWORD *)v7 + 21) )
    {
      v7 += 84;
      break;
    }
    v7 += 112;
    if ( v7 == (const char *)&unk_3F72430 )
    {
      if ( v17[0] != 109 )
        goto LABEL_3;
      goto LABEL_18;
    }
  }
  if ( v7 == "LocationSize::" )
    goto LABEL_3;
LABEL_18:
  if ( v7[4] == (v7[4] & a3)
    && (v9 = *(_QWORD *)(a2 + 24), v10 = *(_QWORD *)(v9 + 16), *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 14)
    && *(_DWORD *)(v9 + 12) - 1 == *((_DWORD *)v7 + 2)
    && ((v11 = *((_DWORD *)v7 + 3), v12 = *((_DWORD *)v7 + 4), v11 < 0)
     || (v15 = (unsigned int)(v11 + 1), sub_BCAC40(*(_QWORD *)(v10 + 8 * v15), 32))
     || sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v15), 64))
    && (v12 < 0
     || (v16 = (unsigned int)(v12 + 1), sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v16), 32))
     || sub_BCAC40(*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v16), 64)) )
  {
    v13 = _mm_loadu_si128((const __m128i *)(v7 + 4));
    v14 = *(_QWORD *)(v7 + 20);
    a1[1].m128i_i8[8] = 1;
    a1[1].m128i_i64[0] = v14;
    *a1 = v13;
  }
  else
  {
LABEL_3:
    a1[1].m128i_i8[8] = 0;
  }
  return a1;
}
