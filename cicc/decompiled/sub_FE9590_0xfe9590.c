// Function: sub_FE9590
// Address: 0xfe9590
//
__int64 __fastcall sub_FE9590(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rcx
  __int64 v4; // rax
  bool v5; // cf
  __int64 result; // rax
  __int16 v7; // cx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int16 v10; // dx
  __m128i v11; // [rsp+0h] [rbp-50h]
  __int64 v12; // [rsp+18h] [rbp-38h] BYREF
  __int64 v13; // [rsp+20h] [rbp-30h] BYREF
  __int16 v14; // [rsp+28h] [rbp-28h]
  __int64 v15; // [rsp+30h] [rbp-20h] BYREF
  __int16 v16; // [rsp+38h] [rbp-18h]

  v2 = *(_QWORD **)(a2 + 128);
  v3 = &v2[*(unsigned int *)(a2 + 136)];
  if ( v3 == v2 )
  {
    v12 = -1;
LABEL_9:
    v9 = sub_FE8600(&v12);
    v15 = 1;
    v13 = v9;
    v14 = v10;
    v16 = 0;
    result = sub_FDE760((__int64)&v15, (__int64)&v13);
    v11 = _mm_loadu_si128((const __m128i *)result);
    v8 = v11.m128i_i64[0];
    v7 = v11.m128i_i16[4];
    goto LABEL_7;
  }
  v4 = 0;
  do
  {
    v5 = __CFADD__(*v2, v4);
    v4 += *v2;
    if ( v5 )
      v4 = -1;
    ++v2;
  }
  while ( v2 != v3 );
  result = ~v4;
  v7 = 12;
  v8 = 1;
  v12 = result;
  if ( result )
    goto LABEL_9;
LABEL_7:
  *(_QWORD *)(a2 + 160) = v8;
  *(_WORD *)(a2 + 168) = v7;
  return result;
}
