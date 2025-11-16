// Function: sub_A5E9B0
// Address: 0xa5e9b0
//
_BYTE *__fastcall sub_A5E9B0(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i *v4; // rdx
  __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int8 v9; // al
  __int64 *v10; // r13
  _BYTE *result; // rax
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+8h] [rbp-38h]
  char *v14; // [rsp+10h] [rbp-30h]
  __int64 v15; // [rsp+18h] [rbp-28h]

  v4 = *(__m128i **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 0xFu )
  {
    sub_CB6200(a1, "!DILexicalBlock(", 16);
  }
  else
  {
    *v4 = _mm_load_si128((const __m128i *)&xmmword_3F24B30);
    *(_QWORD *)(a1 + 32) += 16LL;
  }
  v15 = a3;
  v5 = a2 - 16;
  v14 = ", ";
  v6 = *(_BYTE *)(a2 - 16);
  v12 = a1;
  v13 = 1;
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(a2 - 32);
  else
    v7 = v5 - 8LL * ((v6 >> 2) & 0xF);
  sub_A5CC00((__int64)&v12, "scope", 5u, *(_QWORD *)(v7 + 8), 0);
  v8 = a2;
  if ( *(_BYTE *)a2 != 16 )
  {
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(__int64 **)(a2 - 32);
    else
      v10 = (__int64 *)(v5 - 8LL * ((v9 >> 2) & 0xF));
    v8 = *v10;
  }
  sub_A5CC00((__int64)&v12, "file", 4u, v8, 1);
  sub_A537C0((__int64)&v12, "line", 4u, *(_DWORD *)(a2 + 4), 1);
  sub_A537C0((__int64)&v12, "column", 6u, *(unsigned __int16 *)(a2 + 16), 1);
  result = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == result )
    return (_BYTE *)sub_CB6200(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 32);
  return result;
}
