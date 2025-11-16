// Function: sub_261AA90
// Address: 0x261aa90
//
void __fastcall sub_261AA90(__int64 a1, __int64 *a2)
{
  __int64 *v4; // rdi
  unsigned int v5; // esi
  __int64 *v6; // rdx
  __int64 v7; // r10
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 v11; // rcx

  if ( (__int64 *)a1 != a2 )
  {
    v4 = (__int64 *)(a1 + 16);
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *((_DWORD *)v4 + 2);
        v6 = v4;
        v4 += 2;
        v7 = *(v4 - 2);
        if ( v5 >= *(_DWORD *)(a1 + 8) )
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
        *(_QWORD *)a1 = v7;
        *(_DWORD *)(a1 + 8) = v5;
        if ( a2 == v4 )
          return;
      }
      if ( v5 < *((_DWORD *)v4 - 6) )
      {
        v10 = v4 - 4;
        do
        {
          v11 = *v10;
          v6 = v10;
          v10 -= 2;
          v10[4] = v11;
          *((_DWORD *)v10 + 10) = *((_DWORD *)v10 + 6);
        }
        while ( v5 < *((_DWORD *)v10 + 2) );
      }
      *v6 = v7;
      *((_DWORD *)v6 + 2) = v5;
    }
  }
}
