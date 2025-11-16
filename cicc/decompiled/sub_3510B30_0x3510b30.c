// Function: sub_3510B30
// Address: 0x3510b30
//
void __fastcall sub_3510B30(__int64 a1, unsigned int *a2)
{
  unsigned int *v3; // rsi
  unsigned int v4; // ecx
  unsigned int *v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r10
  __int64 v8; // rax
  unsigned int v9; // ecx
  unsigned int *i; // rax

  if ( (unsigned int *)a1 != a2 )
  {
    v3 = (unsigned int *)(a1 + 16);
    while ( a2 != v3 )
    {
      while ( 1 )
      {
        v4 = *v3;
        v5 = v3;
        v3 += 4;
        v6 = *((_QWORD *)v3 - 1);
        if ( *(_DWORD *)a1 >= v4 )
          break;
        v7 = *((_QWORD *)v3 - 2);
        v8 = ((__int64)v5 - a1) >> 4;
        if ( (__int64)v5 - a1 > 0 )
        {
          do
          {
            v9 = *(v5 - 4);
            v5 -= 4;
            v5[4] = v9;
            *((_QWORD *)v5 + 3) = *((_QWORD *)v5 + 1);
            --v8;
          }
          while ( v8 );
        }
        *(_DWORD *)a1 = v7;
        *(_QWORD *)(a1 + 8) = v6;
        if ( a2 == v3 )
          return;
      }
      for ( i = v3 - 8; v4 > *i; i -= 4 )
      {
        i[4] = *i;
        *((_QWORD *)i + 3) = *((_QWORD *)i + 1);
        v5 = i;
      }
      *v5 = v4;
      *((_QWORD *)v5 + 1) = v6;
    }
  }
}
