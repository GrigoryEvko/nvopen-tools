// Function: sub_1550360
// Address: 0x1550360
//
_BYTE *__fastcall sub_1550360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  unsigned int v11; // ecx
  unsigned __int8 v12; // r14
  __int64 v13; // rdi
  _WORD *v14; // rdx
  _WORD *v15; // rdx
  const char *v16; // rax
  size_t v17; // rdx
  _BYTE *result; // rax
  unsigned __int8 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  char v22; // [rsp+8h] [rbp-58h]
  const char *v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v9 = *(__m128i **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v9 <= 0x11u )
  {
    sub_16E7EE0(a1, "!DISubroutineType(", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F24B20);
    v9[1].m128i_i16[0] = 10341;
    *v9 = si128;
    *(_QWORD *)(a1 + 24) += 18LL;
  }
  v26 = a5;
  v11 = *(_DWORD *)(a2 + 28);
  v25 = a4;
  v21 = a1;
  v22 = 1;
  v23 = ", ";
  v24 = a3;
  sub_154B2B0(&v21, "flags", 5u, v11);
  v12 = *(_BYTE *)(a2 + 52);
  if ( v12 )
  {
    v13 = v21;
    if ( v22 )
      v22 = 0;
    else
      v13 = sub_1263B40(v21, v23);
    v14 = *(_WORD **)(v13 + 24);
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
    {
      v20 = sub_16E7EE0(v13, "cc", 2);
      v15 = *(_WORD **)(v20 + 24);
      v13 = v20;
    }
    else
    {
      *v14 = 25443;
      v15 = (_WORD *)(*(_QWORD *)(v13 + 24) + 2LL);
      *(_QWORD *)(v13 + 24) = v15;
    }
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v15 <= 1u )
    {
      sub_16E7EE0(v13, ": ", 2);
    }
    else
    {
      *v15 = 8250;
      *(_QWORD *)(v13 + 24) += 2LL;
    }
    v16 = sub_14E8990(v12);
    if ( v17 )
    {
      sub_1549FF0(v21, v16, v17);
    }
    else
    {
      v19 = *(unsigned __int8 **)(v21 + 24);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(v21 + 16) )
      {
        sub_16E7DE0(v21, v12);
      }
      else
      {
        *(_QWORD *)(v21 + 24) = v19 + 1;
        *v19 = v12;
      }
    }
  }
  sub_154F950((__int64)&v21, "types", 5u, *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8))), 0);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
