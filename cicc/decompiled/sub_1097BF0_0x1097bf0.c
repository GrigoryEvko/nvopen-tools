// Function: sub_1097BF0
// Address: 0x1097bf0
//
__int64 __fastcall sub_1097BF0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __m128i *v3; // rax
  __m128i si128; // xmm0
  __int64 v6; // rdx
  __int64 v7; // rax
  __m128i *v8; // rax
  __int64 v9; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v11[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = sub_1095C70((_QWORD *)a2);
  if ( *(_BYTE *)(a2 + 129) )
  {
    v9 = 32;
    v10[0] = v11;
    v8 = (__m128i *)sub_22409D0(v10, &v9, 0);
    v10[0] = v8;
    v11[0] = v9;
    *v8 = _mm_load_si128((const __m128i *)&xmmword_3F90130);
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_3F90180);
LABEL_5:
    v10[1] = v9;
    *(_BYTE *)(v10[0] + v9) = 0;
    sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)v10);
    if ( (_QWORD *)v10[0] != v11 )
      j_j___libc_free_0(v10[0], v11[0] + 1LL);
    return a1;
  }
  if ( *(_BYTE *)(a2 + 118) )
  {
    for ( ; v2 != -1; v2 = sub_1095C70((_QWORD *)a2) )
    {
      if ( v2 == 34 )
      {
        if ( (unsigned int)sub_1095CA0((_QWORD *)a2) != 34 )
          goto LABEL_18;
        sub_1095C70((_QWORD *)a2);
      }
    }
LABEL_4:
    v9 = 28;
    v10[0] = v11;
    v3 = (__m128i *)sub_22409D0(v10, &v9, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F90170);
    v10[0] = v3;
    v11[0] = v9;
    qmemcpy(&v3[1], "ing constant", 12);
    *v3 = si128;
    goto LABEL_5;
  }
  while ( v2 != 34 )
  {
    if ( v2 == 92 )
      v2 = sub_1095C70((_QWORD *)a2);
    if ( v2 == -1 )
      goto LABEL_4;
    v2 = sub_1095C70((_QWORD *)a2);
  }
LABEL_18:
  v6 = *(_QWORD *)(a2 + 104);
  v7 = *(_QWORD *)(a2 + 152);
  *(_DWORD *)a1 = 3;
  *(_DWORD *)(a1 + 32) = 64;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 16) = v7 - v6;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
