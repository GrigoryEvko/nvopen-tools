// Function: sub_257BF90
// Address: 0x257bf90
//
__int64 __fastcall sub_257BF90(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v9; // r12d
  __int64 v11; // rax
  __int64 v13; // [rsp+14h] [rbp-3Ch] BYREF
  int v14; // [rsp+1Ch] [rbp-34h]

  *a5 = 0;
  v13 = 0x3300000032LL;
  v14 = 29;
  v9 = sub_2516400(a1, a3, (__int64)&v13, 3, a6, 29);
  if ( (_BYTE)v9 )
  {
    *a5 = 1;
  }
  else if ( a2 )
  {
    v11 = sub_257BB10(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
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
