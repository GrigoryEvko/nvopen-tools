// Function: sub_2E798E0
// Address: 0x2e798e0
//
__m128i *__fastcall sub_2E798E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __m128i *result; // rax
  unsigned int v5; // r14d
  __int64 i; // rbx
  unsigned int v8; // eax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // [rsp+8h] [rbp-38h]

  result = (__m128i *)*(unsigned int *)(a2 + 64);
  LODWORD(v11) = (_DWORD)result;
  if ( (_DWORD)result )
  {
    v5 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    if ( v5 > a4 )
      v5 = a4;
    if ( v5 )
    {
      for ( i = 0; i != v5; ++i )
      {
        HIDWORD(v11) = i;
        result = (__m128i *)(*(_QWORD *)(a2 + 32) + 40 * i);
        if ( !result->m128i_i8[0] && (result->m128i_i8[3] & 0x10) != 0 )
        {
          v8 = sub_2E8E690(a3);
          result = sub_2E79810(a1, v11, ((unsigned __int64)HIDWORD(v11) << 32) | v8, 0, v9, v10);
        }
      }
    }
  }
  return result;
}
