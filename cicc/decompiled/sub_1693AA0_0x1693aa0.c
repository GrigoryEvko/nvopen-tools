// Function: sub_1693AA0
// Address: 0x1693aa0
//
void __fastcall sub_1693AA0(char *a1, char *a2)
{
  char *v4; // rdi
  unsigned __int64 v5; // rsi
  char *v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rcx
  char *i; // rax

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
      for ( i = v4 - 32; v5 < *(_QWORD *)i; i -= 16 )
      {
        *((_QWORD *)i + 2) = *(_QWORD *)i;
        *((_QWORD *)i + 3) = *((_QWORD *)i + 1);
        v6 = i;
      }
      *(_QWORD *)v6 = v5;
      *((_QWORD *)v6 + 1) = v7;
    }
  }
}
