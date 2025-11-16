// Function: sub_1E73C00
// Address: 0x1e73c00
//
__int64 __fastcall sub_1E73C00(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // eax
  _QWORD *v4; // r14
  _QWORD *v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // r12
  __int64 result; // rax
  _QWORD *v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rdi
  __m128i si128; // xmm0
  __int64 v13; // rax
  char *v14; // rdx
  __int64 v15; // rdi
  char *v16; // rax
  bool v17; // cf

  v2 = *(_QWORD *)(a1 + 128);
  if ( (*(_BYTE *)(v2 + 580) & 1) == 0 )
    sub_1F01DD0(v2 + 344);
  v3 = *(_DWORD *)(v2 + 584);
  v4 = *(_QWORD **)(a1 + 584);
  v5 = *(_QWORD **)(a1 + 576);
  *(_DWORD *)(a1 + 32) = v3;
  if ( v5 != v4 )
  {
    while ( 1 )
    {
      v7 = *v5;
      if ( (*(_BYTE *)(*v5 + 236LL) & 1) == 0 )
        break;
      v6 = *(_DWORD *)(v7 + 240);
      if ( *(_DWORD *)(a1 + 32) < v6 )
        goto LABEL_6;
LABEL_7:
      if ( v4 == ++v5 )
        goto LABEL_12;
    }
    sub_1F01DD0(*v5);
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
    if ( v9[2] - (_QWORD)v10 <= 0x16u )
    {
      v11 = sub_16E7EE0((__int64)v9, "Critical Path(GS-RR ): ", 0x17u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EC350);
      v10[1].m128i_i32[0] = 542265901;
      v10[1].m128i_i16[2] = 14889;
      v10[1].m128i_i8[6] = 32;
      *v10 = si128;
      v9[3] += 23LL;
    }
    v13 = sub_16E7A90(v11, *(unsigned int *)(a1 + 32));
    v14 = *(char **)(v13 + 24);
    v15 = v13;
    v16 = *(char **)(v13 + 16);
    v17 = v16 == v14;
    result = v16 - v14;
    if ( v17 || result == 1 )
    {
      result = sub_16E7EE0(v15, " \n", 2u);
    }
    else
    {
      *(_WORD *)v14 = 2592;
      *(_QWORD *)(v15 + 24) += 2LL;
    }
  }
  if ( byte_4FC7AE0 )
  {
    result = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 4LL);
    if ( (_DWORD)result )
    {
      *(_DWORD *)(a1 + 36) = sub_1E716C0(*(_QWORD *)(a1 + 128));
      return sub_1E73BB0(a1);
    }
  }
  return result;
}
