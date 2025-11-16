// Function: sub_25DCCA0
// Address: 0x25dcca0
//
void __fastcall sub_25DCCA0(char *a1, char *a2)
{
  char *v2; // rcx
  unsigned __int64 v5; // rdi
  char *v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r10
  __int64 v9; // rsi
  char *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  char *i; // rax
  __int64 v15; // rdx

  if ( a1 != a2 )
  {
    v2 = a1 + 24;
    while ( a2 != v2 )
    {
      v5 = *((_QWORD *)v2 + 2);
      v6 = v2;
      v2 += 24;
      v7 = *((_QWORD *)v2 - 3);
      v8 = *((_QWORD *)v2 - 2);
      if ( v5 >= *((_QWORD *)a1 + 2) )
      {
        v13 = *((_QWORD *)v2 - 4);
        for ( i = v2 - 48; v5 < v13; v13 = *((_QWORD *)i + 2) )
        {
          *((_QWORD *)i + 5) = v13;
          v15 = *((_QWORD *)i + 1);
          v6 = i;
          i -= 24;
          *((_QWORD *)i + 7) = v15;
          *((_QWORD *)i + 6) = *((_QWORD *)i + 3);
        }
        *((_QWORD *)v6 + 2) = v5;
        *((_QWORD *)v6 + 1) = v8;
        *(_QWORD *)v6 = v7;
      }
      else
      {
        v9 = v6 - a1;
        v10 = v2;
        v11 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 3);
        if ( v9 > 0 )
        {
          do
          {
            v12 = *((_QWORD *)v10 - 4);
            v10 -= 24;
            *((_QWORD *)v10 + 2) = v12;
            *((_QWORD *)v10 + 1) = *((_QWORD *)v10 - 2);
            *(_QWORD *)v10 = *((_QWORD *)v10 - 3);
            --v11;
          }
          while ( v11 );
        }
        *((_QWORD *)a1 + 2) = v5;
        *((_QWORD *)a1 + 1) = v8;
        *(_QWORD *)a1 = v7;
      }
    }
  }
}
