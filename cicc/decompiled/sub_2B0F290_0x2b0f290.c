// Function: sub_2B0F290
// Address: 0x2b0f290
//
void __fastcall sub_2B0F290(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  int v5; // edi
  __int64 v6; // rdx
  int v7; // r9d
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax

  if ( a1 != a2 )
  {
    v4 = a1 + 16;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *(_DWORD *)(v4 + 4);
        v6 = v4;
        v4 += 16;
        v7 = *(_DWORD *)(v4 - 16);
        v8 = *(_QWORD *)(v4 - 8);
        if ( v5 >= *(_DWORD *)(a1 + 4) )
          break;
        v9 = v4;
        v10 = v6 - a1;
        v11 = (v6 - a1) >> 4;
        if ( v10 > 0 )
        {
          do
          {
            v12 = *(_QWORD *)(v9 - 24);
            v9 -= 16;
            *(_QWORD *)(v9 + 8) = v12;
            *(_DWORD *)(v9 + 4) = *(_DWORD *)(v9 - 12);
            *(_DWORD *)v9 = *(_DWORD *)(v9 - 16);
            --v11;
          }
          while ( v11 );
        }
        *(_QWORD *)(a1 + 8) = v8;
        *(_DWORD *)(a1 + 4) = v5;
        *(_DWORD *)a1 = v7;
        if ( a2 == v4 )
          return;
      }
      v13 = v4 - 32;
      if ( v5 < *(_DWORD *)(v4 - 28) )
      {
        do
        {
          *(_QWORD *)(v13 + 24) = *(_QWORD *)(v13 + 8);
          *(_DWORD *)(v13 + 20) = *(_DWORD *)(v13 + 4);
          *(_DWORD *)(v13 + 16) = *(_DWORD *)v13;
          v6 = v13;
          v13 -= 16;
        }
        while ( v5 < *(_DWORD *)(v13 + 4) );
      }
      *(_QWORD *)(v6 + 8) = v8;
      *(_DWORD *)(v6 + 4) = v5;
      *(_DWORD *)v6 = v7;
    }
  }
}
