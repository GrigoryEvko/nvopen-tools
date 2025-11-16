// Function: sub_1DE3CF0
// Address: 0x1de3cf0
//
void __fastcall sub_1DE3CF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rax
  int v8; // ecx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // [rsp-8h] [rbp-8h]

  if ( a1 != a2 )
  {
    v3 = a1 + 16;
    while ( a2 != v3 )
    {
      v4 = *(_DWORD *)(v3 + 8);
      v5 = v3;
      v3 += 16;
      v6 = *(_QWORD *)(v3 - 16);
      if ( *(_DWORD *)(a1 + 8) >= v4 )
      {
        v10 = v3 - 32;
        if ( v4 > *(_DWORD *)(v3 - 24) )
        {
          do
          {
            *(_DWORD *)(v10 + 24) = *(_DWORD *)(v10 + 8);
            *(_QWORD *)(v10 + 16) = *(_QWORD *)v10;
            v5 = v10;
            v10 -= 16;
          }
          while ( v4 > *(_DWORD *)(v10 + 8) );
        }
        *(_DWORD *)(v5 + 8) = v4;
        *(_QWORD *)v5 = v6;
      }
      else
      {
        *((_DWORD *)&v11 - 2) = *(_DWORD *)(v3 - 8);
        v7 = (v5 - a1) >> 4;
        if ( v5 - a1 > 0 )
        {
          do
          {
            v8 = *(_DWORD *)(v5 - 8);
            v5 -= 16;
            *(_DWORD *)(v5 + 24) = v8;
            *(_QWORD *)(v5 + 16) = *(_QWORD *)v5;
            --v7;
          }
          while ( v7 );
        }
        v9 = *((_DWORD *)&v11 - 2);
        *(_QWORD *)a1 = v6;
        *(_DWORD *)(a1 + 8) = v9;
      }
    }
  }
}
