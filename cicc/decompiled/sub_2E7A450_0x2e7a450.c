// Function: sub_2E7A450
// Address: 0x2e7a450
//
__int64 __fastcall sub_2E7A450(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __m128i v5; // xmm0
  __int64 v6; // rax
  signed __int64 v7; // r14
  char *v8; // rcx
  __int64 v9; // r14
  const void *v10; // rsi
  size_t v11; // rax
  __int64 v12; // r12

  v4 = a1[46];
  if ( v4 == a1[47] )
  {
    sub_106F080((__int64)(a1 + 45), (const __m128i *)v4, (__int64 *)a2);
    v12 = a1[46];
  }
  else
  {
    if ( v4 )
    {
      v5 = _mm_loadu_si128((const __m128i *)(a2 + 8));
      *(_QWORD *)v4 = *(_QWORD *)a2;
      v6 = *(_QWORD *)(a2 + 24);
      *(__m128i *)(v4 + 8) = v5;
      *(_QWORD *)(v4 + 24) = v6;
      *(_BYTE *)(v4 + 32) = *(_BYTE *)(a2 + 32);
      *(_QWORD *)(v4 + 40) = *(_QWORD *)(a2 + 40);
      v7 = *(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48);
      *(_QWORD *)(v4 + 48) = 0;
      *(_QWORD *)(v4 + 56) = 0;
      *(_QWORD *)(v4 + 64) = 0;
      if ( v7 )
      {
        if ( v7 < 0 )
          sub_4261EA(a1, a2, a3);
        v8 = (char *)sub_22077B0(v7);
      }
      else
      {
        v8 = 0;
      }
      *(_QWORD *)(v4 + 48) = v8;
      *(_QWORD *)(v4 + 64) = &v8[v7];
      v9 = 0;
      *(_QWORD *)(v4 + 56) = v8;
      v10 = *(const void **)(a2 + 48);
      v11 = *(_QWORD *)(a2 + 56) - (_QWORD)v10;
      if ( v11 )
      {
        v9 = *(_QWORD *)(a2 + 56) - (_QWORD)v10;
        v8 = (char *)memmove(v8, v10, v11);
      }
      *(_QWORD *)(v4 + 56) = &v8[v9];
      *(_QWORD *)(v4 + 72) = v4 + 88;
      sub_2E78280((__int64 *)(v4 + 72), *(_BYTE **)(a2 + 72), *(_QWORD *)(a2 + 72) + *(_QWORD *)(a2 + 80));
      v4 = a1[46];
    }
    v12 = v4 + 104;
    a1[46] = v12;
  }
  return -991146299 * (unsigned int)((v12 - a1[45]) >> 3) - 1;
}
