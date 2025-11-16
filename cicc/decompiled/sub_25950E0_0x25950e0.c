// Function: sub_25950E0
// Address: 0x25950e0
//
__int64 __fastcall sub_25950E0(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v9; // r12d
  __int64 v11; // rax
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  *a5 = 0;
  v13[0] = 0x4C00000013LL;
  v9 = sub_2516400(a1, a3, (__int64)v13, 2, a6, 19);
  if ( (_BYTE)v9 )
  {
    *a5 = 1;
  }
  else if ( a2 )
  {
    v11 = sub_2594C80(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v11;
    if ( v11 )
    {
      v9 = *(unsigned __int8 *)(v11 + 97);
      if ( (_BYTE)v9 )
        *a5 = *(_BYTE *)(v11 + 96);
    }
  }
  return v9;
}
