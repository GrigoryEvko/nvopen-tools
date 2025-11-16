// Function: sub_1DD75B0
// Address: 0x1dd75b0
//
__int64 __fastcall sub_1DD75B0(_QWORD *a1, __int64 a2)
{
  unsigned int *v3; // rax
  unsigned int *v5; // r10
  unsigned int *v6; // rdi
  unsigned int *v7; // rax
  unsigned int v8; // edx
  int v9; // r8d
  unsigned int v10; // ecx
  __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned int v13[3]; // [rsp+Ch] [rbp-14h] BYREF

  if ( a1[15] == a1[14] )
  {
    sub_16AF710(v13, 1u, (__int64)(a1[12] - a1[11]) >> 3);
    return v13[0];
  }
  else
  {
    v3 = (unsigned int *)sub_1DD7590((__int64)a1, a2);
    if ( *v3 == -1 )
    {
      v5 = (unsigned int *)a1[14];
      v6 = (unsigned int *)a1[15];
      if ( v6 == v5 )
      {
        v12 = 0x80000000;
        v9 = 0;
      }
      else
      {
        v7 = (unsigned int *)a1[14];
        v8 = 0;
        v9 = 0;
        do
        {
          v10 = *v7;
          if ( *v7 != -1 )
          {
            v11 = v8;
            v8 += v10;
            if ( (unsigned __int64)v10 + v11 > 0x80000000 )
              v8 = 0x80000000;
            ++v9;
          }
          ++v7;
        }
        while ( v6 != v7 );
        v12 = 0x80000000 - v8;
      }
      return v12 / ((unsigned int)(v6 - v5) - v9);
    }
    else
    {
      return *v3;
    }
  }
}
