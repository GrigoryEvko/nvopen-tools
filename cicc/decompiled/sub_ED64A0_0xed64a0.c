// Function: sub_ED64A0
// Address: 0xed64a0
//
void __fastcall sub_ED64A0(char *a1, char *a2)
{
  char *v4; // rdi
  unsigned __int64 v5; // rsi
  char *v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // rcx
  char *i; // rdx
  __int64 v12; // rax

  if ( a1 != a2 )
  {
    v4 = a1 + 16;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)v4;
        v6 = v4;
        v4 += 16;
        v7 = *((_QWORD *)v4 - 1);
        if ( v5 >= *(_QWORD *)a1 )
          break;
        v8 = (v6 - a1) >> 4;
        if ( v6 - a1 > 0 )
        {
          do
          {
            v9 = *((_QWORD *)v6 - 2);
            v6 -= 16;
            *((_QWORD *)v6 + 2) = v9;
            *((_QWORD *)v6 + 3) = *((_QWORD *)v6 + 1);
            --v8;
          }
          while ( v8 );
        }
        *(_QWORD *)a1 = v5;
        *((_QWORD *)a1 + 1) = v7;
        if ( a2 == v4 )
          return;
      }
      v10 = *((_QWORD *)v4 - 4);
      for ( i = v4 - 32; v5 < v10; i -= 16 )
      {
        v12 = *((_QWORD *)i + 1);
        *((_QWORD *)i + 2) = v10;
        *((_QWORD *)i + 3) = v12;
        v6 = i;
        v10 = *((_QWORD *)i - 2);
      }
      *(_QWORD *)v6 = v5;
      *((_QWORD *)v6 + 1) = v7;
    }
  }
}
