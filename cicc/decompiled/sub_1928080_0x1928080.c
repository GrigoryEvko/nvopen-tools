// Function: sub_1928080
// Address: 0x1928080
//
void __fastcall sub_1928080(unsigned int *a1, int *a2, __int64 a3, __int64 a4)
{
  int *v4; // r13
  char v7; // al
  int *v8; // rdi
  unsigned int v9; // esi
  unsigned int v10; // ecx
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]

  v13 = a3;
  v14 = a4;
  if ( a1 != (unsigned int *)a2 )
  {
    v4 = (int *)(a1 + 2);
    if ( a1 + 2 != (unsigned int *)a2 )
    {
      do
      {
        v7 = sub_1921830(&v13, v4, a1);
        v8 = v4;
        v4 += 2;
        if ( v7 )
        {
          v9 = *(v4 - 2);
          v10 = *(v4 - 1);
          v11 = ((char *)v8 - (char *)a1) >> 3;
          if ( (char *)v8 - (char *)a1 > 0 )
          {
            do
            {
              v12 = *(v8 - 2);
              v8 -= 2;
              v8[2] = v12;
              v8[3] = v8[1];
              --v11;
            }
            while ( v11 );
          }
          *a1 = v9;
          a1[1] = v10;
        }
        else
        {
          sub_1928010((__int64 *)v8, v13, v14);
        }
      }
      while ( a2 != v4 );
    }
  }
}
