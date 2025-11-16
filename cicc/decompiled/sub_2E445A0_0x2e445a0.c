// Function: sub_2E445A0
// Address: 0x2e445a0
//
__int64 __fastcall sub_2E445A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // rdi
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 *v18; // r9
  __int64 *v19; // r12
  __int64 *j; // r15
  __int64 v21; // r8
  __int64 v22; // rcx
  _WORD *v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 i; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  size_t v31; // [rsp+18h] [rbp-38h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x4Du )
  {
    v7 = sub_CB6200(v7, "Printing analysis 'Machine Branch Probability Analysis' for machine function '", 0x4Eu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    qmemcpy(&v8[4], "ine function '", 14);
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_4450080);
    v8[2] = _mm_load_si128((const __m128i *)&xmmword_4450090);
    v8[3] = _mm_load_si128((const __m128i *)&xmmword_44500A0);
    *(_QWORD *)(v7 + 32) += 78LL;
  }
  v10 = sub_2E791E0(a3);
  v12 = *(_BYTE **)(v7 + 32);
  v13 = (unsigned __int8 *)v10;
  v14 = *(_QWORD *)(v7 + 24) - (_QWORD)v12;
  if ( v14 < v11 )
  {
    v27 = sub_CB6200(v7, v13, v11);
    v12 = *(_BYTE **)(v27 + 32);
    v7 = v27;
    v14 = *(_QWORD *)(v27 + 24) - (_QWORD)v12;
  }
  else if ( v11 )
  {
    v31 = v11;
    memcpy(v12, v13, v11);
    v12 = (_BYTE *)(v31 + *(_QWORD *)(v7 + 32));
    v26 = *(_QWORD *)(v7 + 24) - (_QWORD)v12;
    *(_QWORD *)(v7 + 32) = v12;
    if ( v26 > 2 )
      goto LABEL_6;
    goto LABEL_17;
  }
  if ( v14 > 2 )
  {
LABEL_6:
    v12[2] = 10;
    *(_WORD *)v12 = 14887;
    *(_QWORD *)(v7 + 32) += 3LL;
    goto LABEL_7;
  }
LABEL_17:
  sub_CB6200(v7, "':\n", 3u);
LABEL_7:
  v15 = sub_2EB2140(a4, &unk_501F1C0);
  v16 = *(_QWORD *)(a3 + 328);
  v17 = v15 + 8;
  for ( i = a3 + 320; i != v16; v16 = *(_QWORD *)(v16 + 8) )
  {
    v18 = *(__int64 **)(v16 + 112);
    v19 = &v18[*(unsigned int *)(v16 + 120)];
    for ( j = v18; v19 != j; ++j )
    {
      v21 = *a2;
      v22 = *j;
      v23 = *(_WORD **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v23 > 1u )
      {
        *v23 = 8224;
        *(_QWORD *)(v21 + 32) += 2LL;
      }
      else
      {
        v30 = *j;
        v24 = sub_CB6200(*a2, (unsigned __int8 *)"  ", 2u);
        v22 = v30;
        v21 = v24;
      }
      sub_2E44300(v17, v21, v16, v22);
    }
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
