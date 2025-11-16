// Function: sub_8A4360
// Address: 0x8a4360
//
__int64 *__fastcall sub_8A4360(__int64 a1, __int64 *a2, unsigned int *a3, int a4, int a5)
{
  __int64 v8; // rsi
  unsigned int v9; // ebx
  __m128i *v11; // rax
  __int64 *v13; // [rsp+10h] [rbp-40h] BYREF
  __int64 **v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *a2;
  if ( a3[1] == -1 )
  {
    v9 = 1;
    if ( v8 )
    {
      sub_89ED70(a1, v8, v14, &v13);
      goto LABEL_5;
    }
  }
  else
  {
    v9 = *a3;
    if ( v8 )
      goto LABEL_3;
  }
  v11 = sub_8A3C00(a1, 0, 0, 0);
  *a2 = (__int64)v11;
  v8 = (__int64)v11;
LABEL_3:
  sub_89ED70(a1, v8, v14, &v13);
  if ( v9 > 1 )
  {
    do
    {
      --v9;
      sub_89ED80(v14, (__int64 ***)&v13);
    }
    while ( v9 != 1 );
  }
LABEL_5:
  if ( ((_BYTE)v14[0][7] & 0x10) == 0 )
    return v13;
  if ( a5 )
  {
    sub_72F220(&v13);
    return v13;
  }
  return sub_866700(a3, a4, (__int64)v14[0], a4 ^ 1u);
}
