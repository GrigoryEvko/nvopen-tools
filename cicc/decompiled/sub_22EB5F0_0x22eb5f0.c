// Function: sub_22EB5F0
// Address: 0x22eb5f0
//
__int64 __fastcall sub_22EB5F0(__int64 a1, __int64 a2, int a3, const __m128i *a4)
{
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 i; // r12
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  __m128i v12; // xmm0
  unsigned __int8 v13; // al
  _QWORD *v14; // rdx
  __int64 v15; // rax
  int v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __m128i v20; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+30h] [rbp-40h]

  v18 = a1;
  v19 = a4[1].m128i_i64[0];
  result = *(_QWORD *)(a1 + 152);
  v6 = *(_QWORD *)(result + 80);
  v7 = result + 72;
  if ( result + 72 == v6 )
  {
    i = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v6 + 32);
      result = v6 + 24;
      if ( i != v6 + 24 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return result;
      if ( !v6 )
        BUG();
    }
  }
  while ( v7 != v6 )
  {
    v9 = i - 24;
    if ( !i )
      v9 = 0;
    v10 = sub_B10CD0(v9 + 48);
    v11 = v10;
    if ( v10 )
    {
      v12 = _mm_loadu_si128(a4);
      v21 = a4[1].m128i_i64[0];
      v20 = v12;
      v13 = *(_BYTE *)(v10 - 16);
      v14 = (v13 & 2) != 0 ? *(_QWORD **)(v11 - 32) : (_QWORD *)(v11 - 16 - 8LL * ((v13 >> 2) & 0xF));
      result = sub_3143CD0(v18, v11, *v14, v19, a3, (unsigned int)&v20, a2);
      if ( (_BYTE)result )
        break;
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v6 + 32) )
    {
      v15 = v6 - 24;
      if ( !v6 )
        v15 = 0;
      result = v15 + 48;
      if ( i != result )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return result;
      if ( !v6 )
        BUG();
    }
  }
  return result;
}
