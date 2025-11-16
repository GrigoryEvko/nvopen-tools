// Function: sub_12559E0
// Address: 0x12559e0
//
__int64 __fastcall sub_12559E0(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned __int8 *v4; // rbx
  unsigned __int8 *v5; // rdi
  unsigned __int64 v7; // r12
  unsigned __int8 *v8; // rdx
  unsigned int v9; // r15d
  char v10; // si
  int v11; // r9d
  unsigned __int8 v12; // al
  unsigned __int8 *v13; // rsi
  int v14; // eax
  unsigned __int8 *v15; // rax
  int v16; // ecx
  unsigned int v18; // [rsp+28h] [rbp-48h] BYREF
  _BYTE v19[4]; // [rsp+2Ch] [rbp-44h] BYREF
  unsigned __int8 *v20; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v21[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a1;
  v5 = &a1[a2];
  if ( v5 == v4 )
  {
    return a3;
  }
  else
  {
    v7 = a2;
    v8 = v4;
    v9 = a3;
    v10 = 1;
    do
    {
      v11 = *v8;
      v12 = ~*v8;
      if ( (unsigned __int8)(*v8 - 65) <= 0x19u )
        v11 = (unsigned __int8)(*v8 + 32);
      ++v8;
      v10 &= v12 >> 7;
      v9 = 33 * v9 + v11;
    }
    while ( v5 != v8 );
    if ( !v10 )
    {
      v9 = a3;
      if ( a2 )
      {
        do
        {
          v20 = v4;
          v21[0] = (unsigned __int64)&v18;
          sub_F03810(&v20, (char *)&v4[v7], v21, (unsigned __int64)v19, 1);
          v13 = &v4[v7];
          if ( v7 < v20 - v4 )
          {
            v4 += v7;
            v7 = 0;
          }
          else
          {
            v7 -= v20 - v4;
            v4 = v20;
          }
          v14 = 105;
          if ( v18 - 304 > 1 )
            v14 = sub_1255D80(v18, v13);
          v18 = v14;
          v21[0] = (unsigned __int64)v19;
          v20 = (unsigned __int8 *)&v18;
          sub_F035C0((unsigned __int64 *)&v20, (unsigned __int64)v19, v21, (unsigned __int64)&v20, 0);
          if ( (_BYTE *)v21[0] != v19 )
          {
            v15 = v19;
            do
            {
              v16 = *v15++;
              v9 += v16 + 32 * v9;
            }
            while ( (unsigned __int8 *)v21[0] != v15 );
          }
        }
        while ( v7 );
      }
    }
  }
  return v9;
}
