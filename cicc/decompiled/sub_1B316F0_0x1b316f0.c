// Function: sub_1B316F0
// Address: 0x1b316f0
//
void __fastcall sub_1B316F0(__int64 a1, unsigned int *a2)
{
  unsigned int *v4; // rsi
  unsigned int v5; // edi
  unsigned int *v6; // rdx
  __int64 v7; // r9
  unsigned int *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned int v11; // ecx
  unsigned int *i; // rax

  if ( (unsigned int *)a1 != a2 )
  {
    v4 = (unsigned int *)(a1 + 16);
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *v4;
        v6 = v4;
        v4 += 4;
        v7 = *((_QWORD *)v4 - 1);
        if ( v5 >= *(_DWORD *)a1 )
          break;
        v8 = v4;
        v9 = (__int64)v6 - a1;
        v10 = ((__int64)v6 - a1) >> 4;
        if ( v9 > 0 )
        {
          do
          {
            v11 = *(v8 - 8);
            v8 -= 4;
            *v8 = v11;
            *((_QWORD *)v8 + 1) = *((_QWORD *)v8 - 1);
            --v10;
          }
          while ( v10 );
        }
        *(_DWORD *)a1 = v5;
        *(_QWORD *)(a1 + 8) = v7;
        if ( a2 == v4 )
          return;
      }
      for ( i = v4 - 8; v5 < *i; i -= 4 )
      {
        i[4] = *i;
        *((_QWORD *)i + 3) = *((_QWORD *)i + 1);
        v6 = i;
      }
      *v6 = v5;
      *((_QWORD *)v6 + 1) = v7;
    }
  }
}
