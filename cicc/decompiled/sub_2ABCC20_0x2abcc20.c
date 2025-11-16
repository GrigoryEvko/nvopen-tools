// Function: sub_2ABCC20
// Address: 0x2abcc20
//
__int64 __fastcall sub_2ABCC20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // r12
  __int64 result; // rax
  __int64 *v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi

  sub_C8CD80((__int64)a1, (__int64)(a1 + 4), a2, a4, a5, a6);
  v7 = *(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96);
  a1[12] = 0;
  a1[13] = 0;
  a1[14] = 0;
  if ( v7 )
  {
    if ( v7 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a1 + 4, v6);
    result = sub_22077B0(v7);
  }
  else
  {
    v7 = 0;
    result = 0;
  }
  a1[12] = result;
  a1[13] = result;
  a1[14] = result + v7;
  v9 = *(__int64 **)(a2 + 104);
  v10 = *(__int64 **)(a2 + 96);
  if ( v9 == v10 )
  {
    a1[13] = result;
  }
  else
  {
    v11 = result + (char *)v9 - (char *)v10;
    do
    {
      if ( result )
      {
        v12 = *v10;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)result = v12;
        if ( *((_BYTE *)v10 + 24) )
        {
          *(_QWORD *)(result + 8) = v10[1];
          v13 = v10[2];
          *(_BYTE *)(result + 24) = 1;
          *(_QWORD *)(result + 16) = v13;
        }
      }
      result += 32;
      v10 += 4;
    }
    while ( result != v11 );
    a1[13] = v11;
  }
  return result;
}
