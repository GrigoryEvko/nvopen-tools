// Function: sub_1550770
// Address: 0x1550770
//
_BYTE *__fastcall sub_1550770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r8
  unsigned __int8 *v14; // rcx
  _BYTE *result; // rax
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  char v17; // [rsp+8h] [rbp-58h]
  char *v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21; // [rsp+28h] [rbp-38h]

  v8 = *(__m128i **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v8 <= 0x13u )
  {
    sub_16E7EE0(a1, "!DIFortranArrayType(", 20);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4293130);
    v8[1].m128i_i32[0] = 677736569;
    *v8 = si128;
    *(_QWORD *)(a1 + 24) += 20LL;
  }
  v21 = a5;
  v18 = ", ";
  v16 = a1;
  v17 = 1;
  v19 = a3;
  v20 = a4;
  sub_1549850(&v16, a2);
  v10 = *(unsigned int *)(a2 + 8);
  v11 = *(_QWORD *)(a2 + 8 * (2 - v10));
  if ( v11 )
  {
    v11 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v10)));
    v13 = v12;
  }
  else
  {
    v13 = 0;
  }
  sub_154AC80(&v16, "name", 4u, v11, v13, 1);
  sub_154F950((__int64)&v16, "scope", 5u, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), 1);
  v14 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)a2 != 15 )
    v14 = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  sub_154F950((__int64)&v16, "file", 4u, v14, 1);
  sub_154ADE0((__int64)&v16, "line", 4u, *(_DWORD *)(a2 + 24), 1);
  sub_154F950((__int64)&v16, "baseType", 8u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 1);
  sub_154B000((__int64)&v16, "size", 4u, *(_QWORD *)(a2 + 32), 1);
  sub_154ADE0((__int64)&v16, "align", 5u, *(_DWORD *)(a2 + 48), 1);
  sub_154B000((__int64)&v16, "offset", 6u, *(_QWORD *)(a2 + 40), 1);
  sub_154B2B0(&v16, "flags", 5u, *(_DWORD *)(a2 + 28));
  sub_154F950((__int64)&v16, "elements", 8u, *(unsigned __int8 **)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))), 1);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
