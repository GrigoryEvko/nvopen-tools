// Function: sub_131D620
// Address: 0x131d620
//
__int64 __fastcall sub_131D620(__int64 a1, __int64 a2, __int64 a3, __m128i *a4, __int64 *a5, _QWORD *a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r9
  __int8 *v12; // rcx
  __int8 *v13; // rsi
  unsigned int v14; // ecx
  unsigned int v15; // ecx
  unsigned int v16; // eax
  __int64 v17; // rdi
  __m128i v18; // [rsp+0h] [rbp-10h] BYREF

  v7 = *(_QWORD *)(a1 + 240);
  v18 = *(__m128i *)(a1 + 240);
  if ( !a4 || !a5 )
  {
LABEL_5:
    result = 0;
    if ( a6 )
    {
      result = 22;
      if ( a7 == 16 )
      {
        v10 = a6[1];
        *(_QWORD *)(a1 + 240) = *a6;
        *(_QWORD *)(a1 + 248) = v10;
        return 0;
      }
    }
    return result;
  }
  v8 = *a5;
  if ( *a5 == 16 )
  {
    *a4 = _mm_loadu_si128(&v18);
    goto LABEL_5;
  }
  if ( (unsigned __int64)*a5 > 0x10 )
    v8 = 16;
  if ( (unsigned int)v8 >= 8 )
  {
    a4->m128i_i64[0] = v7;
    v11 = (unsigned __int64)&a4->m128i_u64[1] & 0xFFFFFFFFFFFFFFF8LL;
    *(__int64 *)((char *)&a4->m128i_i64[-1] + (unsigned int)v8) = *(__int64 *)((char *)&v18.m128i_i64[-1]
                                                                             + (unsigned int)v8);
    v12 = &a4->m128i_i8[-v11];
    v13 = (__int8 *)((char *)&v18 - v12);
    v14 = (v8 + (_DWORD)v12) & 0xFFFFFFF8;
    if ( v14 >= 8 )
    {
      v15 = v14 & 0xFFFFFFF8;
      v16 = 0;
      do
      {
        v17 = v16;
        v16 += 8;
        *(_QWORD *)(v11 + v17) = *(_QWORD *)&v13[v17];
      }
      while ( v16 < v15 );
    }
  }
  else if ( (v8 & 4) != 0 )
  {
    a4->m128i_i32[0] = v18.m128i_i32[0];
    *(__int32 *)((char *)&a4->m128i_i32[-1] + (unsigned int)v8) = *(__int32 *)((char *)&v18.m128i_i32[-1]
                                                                             + (unsigned int)v8);
  }
  else if ( (_DWORD)v8 )
  {
    a4->m128i_i8[0] = v18.m128i_i8[0];
    if ( (v8 & 2) != 0 )
      *(__int16 *)((char *)&a4->m128i_i16[-1] + (unsigned int)v8) = *(__int16 *)((char *)&v18.m128i_i16[-1]
                                                                               + (unsigned int)v8);
  }
  *a5 = v8;
  return 22;
}
