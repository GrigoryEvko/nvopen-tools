// Function: sub_2472230
// Address: 0x2472230
//
__int64 __fastcall sub_2472230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  const __m128i *v13; // rdx
  unsigned __int64 v14; // r13
  __int64 v15; // rdi
  const void *v16; // rsi
  _QWORD v17[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (_BYTE)qword_4FE7EA8 )
  {
    result = sub_246F3F0(a1, a2);
    v6 = result;
    if ( !result )
      return result;
    result = sub_246EE10(a1, a2);
  }
  else
  {
    result = sub_246F3F0(a1, a2);
    v6 = result;
    if ( !result || *(_BYTE *)result <= 0x1Cu )
      return result;
    result = sub_246EE10(a1, a2);
    if ( result && *(_BYTE *)result <= 0x1Cu )
      result = 0;
  }
  if ( *(_BYTE *)(a1 + 632) )
  {
    v17[1] = result;
    v9 = *(unsigned int *)(a1 + 648);
    v10 = *(unsigned int *)(a1 + 652);
    v17[0] = v6;
    v11 = v9 + 1;
    v17[2] = a3;
    if ( v9 + 1 > v10 )
    {
      v14 = *(_QWORD *)(a1 + 640);
      v15 = a1 + 640;
      v16 = (const void *)(a1 + 656);
      if ( v14 <= (unsigned __int64)v17 && (unsigned __int64)v17 < v14 + 24 * v9 )
      {
        sub_C8D5F0(v15, v16, v11, 0x18u, v7, v8);
        v12 = *(_QWORD *)(a1 + 640);
        v9 = *(unsigned int *)(a1 + 648);
        v13 = (const __m128i *)((char *)v17 + v12 - v14);
      }
      else
      {
        sub_C8D5F0(v15, v16, v11, 0x18u, v7, v8);
        v12 = *(_QWORD *)(a1 + 640);
        v9 = *(unsigned int *)(a1 + 648);
        v13 = (const __m128i *)v17;
      }
    }
    else
    {
      v12 = *(_QWORD *)(a1 + 640);
      v13 = (const __m128i *)v17;
    }
    result = v12 + 24 * v9;
    *(__m128i *)result = _mm_loadu_si128(v13);
    *(_QWORD *)(result + 16) = v13[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 648);
  }
  return result;
}
