// Function: sub_1E6E190
// Address: 0x1e6e190
//
__int64 __fastcall sub_1E6E190(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // eax
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  unsigned int v6; // eax
  __int64 v7; // r12
  __int64 result; // rax
  _QWORD *v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rdi
  __m128i si128; // xmm0
  __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // rdi

  v2 = *(_QWORD *)(a1 + 128);
  if ( (*(_BYTE *)(v2 + 580) & 1) == 0 )
    sub_1F01DD0(v2 + 344);
  v3 = *(_DWORD *)(v2 + 584);
  v4 = *(_QWORD **)(a1 + 504);
  *(_DWORD *)(a1 + 32) = v3;
  v5 = &v4[*(unsigned int *)(a1 + 512)];
  if ( v5 != v4 )
  {
    while ( 1 )
    {
      v7 = *v4;
      if ( (*(_BYTE *)(*v4 + 236LL) & 1) == 0 )
        break;
      v6 = *(_DWORD *)(v7 + 240);
      if ( *(_DWORD *)(a1 + 32) < v6 )
        goto LABEL_6;
LABEL_7:
      if ( v5 == ++v4 )
        goto LABEL_12;
    }
    sub_1F01DD0(*v4);
    v6 = *(_DWORD *)(v7 + 240);
    if ( *(_DWORD *)(a1 + 32) >= v6 )
      goto LABEL_7;
    if ( (*(_BYTE *)(v7 + 236) & 1) == 0 )
    {
      sub_1F01DD0(v7);
      v6 = *(_DWORD *)(v7 + 240);
    }
LABEL_6:
    *(_DWORD *)(a1 + 32) = v6;
    goto LABEL_7;
  }
LABEL_12:
  result = (__int64)qword_4FC7CE0;
  if ( LOBYTE(qword_4FC7CE0[20]) )
  {
    v9 = sub_16E8CB0();
    v10 = (__m128i *)v9[3];
    v11 = (__int64)v9;
    if ( v9[2] - (_QWORD)v10 <= 0x17u )
    {
      v11 = sub_16E7EE0((__int64)v9, "Critical Path(PGS-RR ): ", 0x18u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EC340);
      v10[1].m128i_i64[0] = 0x203A292052522D53LL;
      *v10 = si128;
      v9[3] += 24LL;
    }
    v13 = sub_16E7A90(v11, *(unsigned int *)(a1 + 32));
    v14 = *(_WORD **)(v13 + 24);
    v15 = v13;
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
    {
      return sub_16E7EE0(v13, " \n", 2u);
    }
    else
    {
      *v14 = 2592;
      *(_QWORD *)(v15 + 24) += 2LL;
      return 2592;
    }
  }
  return result;
}
