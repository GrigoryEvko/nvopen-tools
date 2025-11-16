// Function: sub_EAA500
// Address: 0xeaa500
//
__int64 __fastcall sub_EAA500(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rcx
  _BOOL4 v6; // r15d
  __m128i *v7; // rbx
  size_t v8; // r8
  size_t v9; // rbx
  size_t v10; // rdx
  unsigned int v11; // eax
  size_t v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]

  result = sub_EAA3A0(a1, a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = a1 + 8;
    v6 = 1;
    if ( !result && v3 != v5 )
    {
      v8 = *(_QWORD *)(a2 + 8);
      v10 = *(_QWORD *)(v3 + 40);
      v9 = v10;
      if ( v8 <= v10 )
        v10 = *(_QWORD *)(a2 + 8);
      if ( v10
        && (v12 = *(_QWORD *)(a2 + 8),
            v11 = memcmp(*(const void **)a2, *(const void **)(v4 + 32), v10),
            v5 = a1 + 8,
            v8 = v12,
            v11) )
      {
        v6 = v11 >> 31;
      }
      else
      {
        v6 = v8 < v9;
        if ( v8 == v9 )
          v6 = 0;
      }
    }
    v13 = v5;
    v7 = (__m128i *)sub_22077B0(48);
    v7[2] = _mm_loadu_si128((const __m128i *)a2);
    sub_220F040(v6, v7, v4, v13);
    ++*(_QWORD *)(a1 + 40);
    return (__int64)v7;
  }
  return result;
}
