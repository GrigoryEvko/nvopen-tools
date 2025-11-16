// Function: sub_33E24E0
// Address: 0x33e24e0
//
__int64 (*__fastcall sub_33E24E0(__int64 a1, unsigned int a2, __m128i *a3, __int64 a4))(void)
{
  __int64 (*result)(void); // rax
  char v7; // r14
  char v8; // r13
  char v9; // al
  __m128i v10; // xmm0
  __m128i v11; // xmm0
  char v12; // [rsp+Fh] [rbp-51h]

  result = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 1360LL);
  if ( (char *)result != (char *)sub_2FE3400 )
  {
    result = (__int64 (*)(void))result();
    if ( !(_BYTE)result )
      return result;
LABEL_9:
    v7 = sub_33E2390(a1, a3->m128i_i64[0], a3->m128i_u64[1], 1);
    v8 = sub_33E2390(a1, *(_QWORD *)a4, *(_QWORD *)(a4 + 8), 1);
    v12 = sub_33E2470(a1, a3->m128i_i64[0]);
    v9 = sub_33E2470(a1, *(_QWORD *)a4);
    if ( v8 != 1 && v7 || v9 != 1 && v12 )
    {
      v10 = _mm_loadu_si128(a3);
      a3->m128i_i64[0] = *(_QWORD *)a4;
      a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
      *(_QWORD *)a4 = v10.m128i_i64[0];
      *(_DWORD *)(a4 + 8) = v10.m128i_i32[2];
      return (__int64 (*)(void))v10.m128i_u32[2];
    }
    else
    {
      result = (__int64 (*)(void))a3->m128i_i64[0];
      if ( *(_DWORD *)(a3->m128i_i64[0] + 24) == 168 )
      {
        result = *(__int64 (**)(void))a4;
        if ( *(_DWORD *)(*(_QWORD *)a4 + 24LL) == 170 )
        {
          v11 = _mm_loadu_si128(a3);
          a3->m128i_i64[0] = (__int64)result;
          a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
          *(_QWORD *)a4 = v11.m128i_i64[0];
          *(_DWORD *)(a4 + 8) = v11.m128i_i32[2];
          return (__int64 (*)(void))v11.m128i_u32[2];
        }
      }
    }
    return result;
  }
  if ( a2 > 0x62 )
  {
    if ( a2 > 0xBC )
    {
      if ( a2 - 279 > 7 )
        return result;
    }
    else if ( a2 <= 0xB9 && a2 - 172 > 0xB )
    {
      return result;
    }
    goto LABEL_9;
  }
  if ( a2 > 0x37 )
  {
    switch ( a2 )
    {
      case '8':
      case ':':
      case '?':
      case '@':
      case 'D':
      case 'F':
      case 'L':
      case 'M':
      case 'R':
      case 'S':
      case '`':
      case 'b':
        goto LABEL_9;
      default:
        return result;
    }
  }
  return result;
}
