// Function: sub_2B0E6E0
// Address: 0x2b0e6e0
//
void __fastcall sub_2B0E6E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned int v5; // edi
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax

  if ( a1 != a2 )
  {
    v4 = a1 + 16;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *(_DWORD *)(v4 + 8);
        v6 = v4;
        v4 += 16;
        v7 = *(_QWORD *)(v4 - 16);
        if ( v5 >= *(_DWORD *)(a1 + 8) )
          break;
        v8 = v4;
        v9 = v6 - a1;
        v10 = (v6 - a1) >> 4;
        if ( v9 > 0 )
        {
          do
          {
            v11 = *(_QWORD *)(v8 - 32);
            v8 -= 16;
            *(_QWORD *)v8 = v11;
            *(_DWORD *)(v8 + 8) = *(_DWORD *)(v8 - 8);
            --v10;
          }
          while ( v10 );
        }
        *(_QWORD *)a1 = v7;
        *(_DWORD *)(a1 + 8) = v5;
        if ( a2 == v4 )
          return;
      }
      v12 = v4 - 32;
      if ( v5 < *(_DWORD *)(v4 - 24) )
      {
        do
        {
          *(_QWORD *)(v12 + 16) = *(_QWORD *)v12;
          *(_DWORD *)(v12 + 24) = *(_DWORD *)(v12 + 8);
          v6 = v12;
          v12 -= 16;
        }
        while ( v5 < *(_DWORD *)(v12 + 8) );
      }
      *(_QWORD *)v6 = v7;
      *(_DWORD *)(v6 + 8) = v5;
    }
  }
}
