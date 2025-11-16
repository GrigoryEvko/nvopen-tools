// Function: sub_2EC3440
// Address: 0x2ec3440
//
__int64 __fastcall sub_2EC3440(__int64 a1)
{
  __int64 v2; // rbx
  __int64 result; // rax
  _QWORD *v4; // r14
  _QWORD *v5; // rbx
  __int64 v6; // r12
  _QWORD *v7; // rax
  __m128i *v8; // rdx
  __int64 v9; // rdi
  __m128i si128; // xmm0
  __int64 v11; // rax
  _WORD *v12; // rdx

  v2 = *(_QWORD *)(a1 + 136);
  if ( (*(_BYTE *)(v2 + 582) & 1) == 0 )
    sub_2F8F5D0(v2 + 328);
  result = *(unsigned int *)(v2 + 568);
  v4 = *(_QWORD **)(a1 + 936);
  v5 = *(_QWORD **)(a1 + 928);
  *(_DWORD *)(a1 + 40) = result;
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      v6 = *v5;
      if ( (*(_BYTE *)(*v5 + 254LL) & 1) == 0 )
        break;
      result = *(unsigned int *)(v6 + 240);
      if ( *(_DWORD *)(a1 + 40) < (unsigned int)result )
        goto LABEL_6;
LABEL_7:
      if ( v4 == ++v5 )
        goto LABEL_12;
    }
    sub_2F8F5D0(*v5);
    result = *(unsigned int *)(v6 + 240);
    if ( *(_DWORD *)(a1 + 40) >= (unsigned int)result )
      goto LABEL_7;
    if ( (*(_BYTE *)(v6 + 254) & 1) == 0 )
    {
      sub_2F8F5D0(v6);
      result = *(unsigned int *)(v6 + 240);
    }
LABEL_6:
    *(_DWORD *)(a1 + 40) = result;
    goto LABEL_7;
  }
LABEL_12:
  if ( (_BYTE)qword_5021808 )
  {
    v7 = sub_CB72A0();
    v8 = (__m128i *)v7[4];
    v9 = (__int64)v7;
    if ( v7[3] - (_QWORD)v8 <= 0x17u )
    {
      v9 = sub_CB6200((__int64)v7, "Critical Path(PGS-RR ): ", 0x18u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EC340);
      v8[1].m128i_i64[0] = 0x203A292052522D53LL;
      *v8 = si128;
      v7[4] += 24LL;
    }
    v11 = sub_CB59D0(v9, *(unsigned int *)(a1 + 40));
    v12 = *(_WORD **)(v11 + 32);
    if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 1u )
    {
      return sub_CB6200(v11, (unsigned __int8 *)" \n", 2u);
    }
    else
    {
      *v12 = 2592;
      *(_QWORD *)(v11 + 32) += 2LL;
      return 2592;
    }
  }
  return result;
}
