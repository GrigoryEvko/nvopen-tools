// Function: sub_214C940
// Address: 0x214c940
//
__int64 __fastcall sub_214C940(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 result; // rax
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  char *v12; // [rsp+10h] [rbp-40h] BYREF
  size_t v13; // [rsp+18h] [rbp-38h]
  _QWORD v14[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = a2;
  v12 = (char *)v14;
  v13 = 0;
  LOBYTE(v14[0]) = 0;
  if ( (unsigned __int8)sub_1C2FA30(a1, &v11) )
  {
    v3 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x13u )
    {
      v2 = sub_16E7EE0(a2, ".attribute(.unified(", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43270F0);
      v3[1].m128i_i32[0] = 677668201;
      *v3 = si128;
      *(_QWORD *)(a2 + 24) += 20LL;
    }
    v5 = sub_16E7A90(v2, v11);
LABEL_5:
    v6 = *(_QWORD *)(v5 + 24);
    v7 = v5;
    if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 2 )
    {
      result = sub_16E7EE0(v5, ")) ", 3u);
    }
    else
    {
      result = 10537;
      *(_BYTE *)(v6 + 2) = 32;
      *(_WORD *)v6 = 10537;
      *(_QWORD *)(v7 + 24) += 3LL;
    }
    goto LABEL_12;
  }
  if ( (unsigned __int8)sub_1C2F120(a1, &v12) )
  {
    v9 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 0x13u )
    {
      v2 = sub_16E7EE0(a2, ".attribute(.unified(", 0x14u);
    }
    else
    {
      v10 = _mm_load_si128((const __m128i *)&xmmword_43270F0);
      v9[1].m128i_i32[0] = 677668201;
      *v9 = v10;
      *(_QWORD *)(a2 + 24) += 20LL;
    }
    v5 = sub_16E7EE0(v2, v12, v13);
    goto LABEL_5;
  }
  result = sub_1263B40(a2, ".attribute(.unified) ");
LABEL_12:
  if ( v12 != (char *)v14 )
    return j_j___libc_free_0(v12, v14[0] + 1LL);
  return result;
}
