// Function: sub_39F4940
// Address: 0x39f4940
//
__int64 __fastcall sub_39F4940(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned __int64 v5; // rbx
  unsigned __int8 v6; // dl
  unsigned __int8 *v8; // rsi
  int v9; // eax
  unsigned __int8 *v10; // rax
  int v11; // ecx
  int v12; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v13; // [rsp+1Ch] [rbp-44h] BYREF
  unsigned __int8 *v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a2;
  while ( v5 )
  {
    v6 = *a1;
    if ( (*a1 & 0x80u) != 0 )
    {
      v14 = a1;
      v15[0] = (unsigned __int64)&v13;
      sub_16F0F70(&v14, (char *)&a1[v5], v15, (unsigned __int64)&v14, 1);
      v8 = &a1[v5];
      if ( v14 - a1 > v5 )
      {
        a1 += v5;
        v5 = 0;
      }
      else
      {
        v5 -= v14 - a1;
        a1 = v14;
      }
      v9 = 105;
      if ( v13 - 304 > 1 )
        v9 = sub_39F4A80(v13, v8);
      v12 = v9;
      v14 = (unsigned __int8 *)&v12;
      v15[0] = (unsigned __int64)&v13;
      sub_16F0D40((unsigned __int64 *)&v14, (unsigned __int64)&v13, v15, (unsigned __int64)&v14, 0);
      if ( (unsigned int *)v15[0] != &v13 )
      {
        v10 = (unsigned __int8 *)&v13;
        do
        {
          v11 = *v10++;
          a3 += v11 + 32 * a3;
        }
        while ( (unsigned __int8 *)v15[0] != v10 );
      }
    }
    else
    {
      if ( (unsigned __int8)(v6 - 65) < 0x1Au )
        v6 += 32;
      --v5;
      ++a1;
      a3 = v6 + 33 * a3;
    }
  }
  return a3;
}
