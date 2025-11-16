// Function: sub_8413E0
// Address: 0x8413e0
//
__int64 __fastcall sub_8413E0(
        __m128i *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        const __m128i **a7)
{
  _BOOL4 v9; // r15d
  const __m128i *v10; // r14
  __int64 result; // rax

  *a6 = 0;
  v9 = (a3 & 0x400) == 0;
  v10 = (const __m128i *)sub_8D46C0(a2);
  if ( !(unsigned int)sub_8D3070(a2) )
  {
LABEL_6:
    if ( !(unsigned int)sub_8D3A70(v10)
      || !(unsigned int)sub_8D4D20(a2)
      || (unsigned int)sub_8DAA20(a1->m128i_i64[0], v10)
      || (const __m128i *)a1->m128i_i64[0] == v10
      || (unsigned int)sub_8DED30(a1->m128i_i64[0], v10, 3) )
    {
      result = sub_840360(a1->m128i_i64, (__int64)v10, 0, 0, v9, v9, a2, 1, a3, a5, a6, a7);
    }
    else
    {
      result = sub_836C50(a1, 0, v10, 0, 1u, 1u, a2, 1, a3, a5, 0, a6, (__int64 **)a7);
    }
    if ( (_DWORD)result )
      goto LABEL_9;
    return 0;
  }
  result = sub_840360(a1->m128i_i64, (__int64)v10, 0, 1, v9, v9, a2, 1, a3, a5, a6, a7);
  if ( !(_DWORD)result )
  {
    if ( *a6 || !(unsigned int)sub_8D4D20(a2) )
      return 0;
    goto LABEL_6;
  }
LABEL_9:
  *(_BYTE *)(a5 + 16) |= 0x20u;
  return result;
}
