// Function: sub_1E40950
// Address: 0x1e40950
//
void __fastcall sub_1E40950(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v4; // rdi
  unsigned __int64 v5; // rsi
  unsigned __int64 *v6; // rax
  int v7; // r9d
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rcx
  unsigned __int64 *i; // rdx
  int v12; // eax

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v4 = (unsigned __int64 *)(a1 + 16);
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *v4;
        v6 = v4;
        v4 += 2;
        v7 = *((_DWORD *)v4 - 2);
        if ( v5 >= *(_QWORD *)a1 )
          break;
        v8 = ((__int64)v6 - a1) >> 4;
        if ( (__int64)v6 - a1 > 0 )
        {
          do
          {
            v9 = *(v6 - 2);
            v6 -= 2;
            v6[2] = v9;
            *((_DWORD *)v6 + 6) = *((_DWORD *)v6 + 2);
            --v8;
          }
          while ( v8 );
        }
        *(_QWORD *)a1 = v5;
        *(_DWORD *)(a1 + 8) = v7;
        if ( a2 == v4 )
          return;
      }
      v10 = *(v4 - 4);
      for ( i = v4 - 4; v5 < v10; i -= 2 )
      {
        v12 = *((_DWORD *)i + 2);
        i[2] = v10;
        *((_DWORD *)i + 6) = v12;
        v6 = i;
        v10 = *(i - 2);
      }
      *v6 = v5;
      *((_DWORD *)v6 + 2) = v7;
    }
  }
}
