// Function: sub_31141F0
// Address: 0x31141f0
//
void __fastcall sub_31141F0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v4; // rsi
  unsigned __int64 *v5; // r9
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r10
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r11
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned __int64 *v14; // r11

  if ( a1 != a2 )
  {
    v4 = a1 + 2;
    if ( a2 != a1 + 2 )
    {
      v5 = a1 + 4;
      do
      {
        while ( 1 )
        {
          v6 = *v4;
          v7 = v4[1];
          v8 = v4;
          if ( *v4 >= *a1 && (*v4 != *a1 || a1[1] <= v7) )
            break;
          v9 = v5;
          v10 = ((char *)v4 - (char *)a1) >> 4;
          if ( (char *)v4 - (char *)a1 > 0 )
          {
            do
            {
              v11 = *(v8 - 2);
              v8 -= 2;
              v8[2] = v11;
              v8[3] = v8[1];
              --v10;
            }
            while ( v10 );
          }
          *a1 = v6;
          v4 += 2;
          v5 += 2;
          a1[1] = v7;
          if ( a2 == v9 )
            return;
        }
        while ( 1 )
        {
          v13 = *(v8 - 2);
          if ( v6 >= v13 && (v6 != v13 || *(v8 - 1) <= v7) )
            break;
          *v8 = v13;
          v12 = *(v8 - 1);
          v8 -= 2;
          v8[3] = v12;
        }
        v14 = v5;
        *v8 = v6;
        v4 += 2;
        v5 += 2;
        v8[1] = v7;
      }
      while ( a2 != v14 );
    }
  }
}
