// Function: sub_8A4DE0
// Address: 0x8a4de0
//
__int64 **__fastcall sub_8A4DE0(__int64 a1, __m128i *a2, __int64 a3, __int64 *a4, int a5, int *a6, __m128i *a7)
{
  __int64 **v8; // rax
  __int64 **v9; // r12

  v8 = sub_8A2270(a1, a2, a3, a4, a5 | 0x200u, a6, a7);
  v9 = v8;
  if ( (__int64 **)a1 != v8
    && ((unsigned int)sub_8D3410(v8) || (unsigned int)sub_8D2310(v9) || !dword_4F077BC && (unsigned int)sub_8D5830(v9)) )
  {
    *a6 = 1;
  }
  return v9;
}
