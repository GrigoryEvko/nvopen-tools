// Function: sub_B8F7D0
// Address: 0xb8f7d0
//
void __fastcall sub_B8F7D0(__int64 a1, unsigned int *a2)
{
  unsigned int *v4; // rsi
  unsigned int v5; // edi
  unsigned int *v6; // rcx
  __int64 v7; // r9
  __int64 v8; // rcx
  unsigned int *v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ecx
  unsigned int v12; // edx
  unsigned int *v13; // rax
  __int64 v14; // rdx

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
        v8 = (__int64)v6 - a1;
        v9 = v4;
        v10 = v8 >> 4;
        if ( v8 > 0 )
        {
          do
          {
            v11 = *(v9 - 8);
            v9 -= 4;
            *v9 = v11;
            *((_QWORD *)v9 + 1) = *((_QWORD *)v9 - 1);
            --v10;
          }
          while ( v10 );
        }
        *(_DWORD *)a1 = v5;
        *(_QWORD *)(a1 + 8) = v7;
        if ( a2 == v4 )
          return;
      }
      v12 = *(v4 - 8);
      v13 = v4 - 8;
      if ( v5 < v12 )
      {
        do
        {
          v13[4] = v12;
          v14 = *((_QWORD *)v13 + 1);
          v6 = v13;
          v13 -= 4;
          *((_QWORD *)v13 + 5) = v14;
          v12 = *v13;
        }
        while ( v5 < *v13 );
      }
      *v6 = v5;
      *((_QWORD *)v6 + 1) = v7;
    }
  }
}
