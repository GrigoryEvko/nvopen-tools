// Function: sub_A5DBD0
// Address: 0xa5dbd0
//
_BYTE *__fastcall sub_A5DBD0(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  unsigned int v7; // ecx
  unsigned __int8 v8; // r14
  __int64 v9; // rdi
  _WORD *v10; // rdx
  const void *v11; // rax
  size_t v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // rbx
  _BYTE *result; // rax
  unsigned __int8 *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+8h] [rbp-48h]
  const char *v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v5 = *(__m128i **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0x11u )
  {
    sub_CB6200(a1, "!DISubroutineType(", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F24B20);
    v5[1].m128i_i16[0] = 10341;
    *v5 = si128;
    *(_QWORD *)(a1 + 32) += 18LL;
  }
  v20 = a3;
  v7 = *(_DWORD *)(a2 + 20);
  v17 = a1;
  v18 = 1;
  v19 = ", ";
  sub_A53C60(&v17, "flags", 5u, v7);
  v8 = *(_BYTE *)(a2 + 44);
  if ( v8 )
  {
    v9 = v17;
    if ( v18 )
      v18 = 0;
    else
      v9 = sub_904010(v17, v19);
    v10 = *(_WORD **)(v9 + 32);
    if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
    {
      v9 = sub_CB6200(v9, "cc", 2);
    }
    else
    {
      *v10 = 25443;
      *(_QWORD *)(v9 + 32) += 2LL;
    }
    sub_904010(v9, ": ");
    v11 = (const void *)sub_E0BA50(v8);
    if ( v12 )
    {
      sub_A51340(v17, v11, v12);
    }
    else
    {
      v16 = *(unsigned __int8 **)(v17 + 32);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v17 + 24) )
      {
        sub_CB5D20(v17, v8);
      }
      else
      {
        *(_QWORD *)(v17 + 32) = v16 + 1;
        *v16 = v8;
      }
    }
  }
  v13 = *(_BYTE *)(a2 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(a2 - 32);
  else
    v14 = a2 - 16 - 8LL * ((v13 >> 2) & 0xF);
  sub_A5CC00((__int64)&v17, "types", 5u, *(_QWORD *)(v14 + 24), 0);
  result = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == result )
    return (_BYTE *)sub_CB6200(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 32);
  return result;
}
