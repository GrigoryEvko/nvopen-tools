// Function: sub_148B340
// Address: 0x148b340
//
char __fastcall sub_148B340(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  char result; // al
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r8
  char v13; // al
  __int64 v14; // [rsp-40h] [rbp-40h]

  result = *(_BYTE *)(a1 + 32);
  if ( result )
  {
    v9 = a2 + 40;
    v10 = *(_QWORD *)(a2 + 48);
    if ( a2 + 40 != v10 )
    {
      do
      {
        if ( !v10 )
          BUG();
        if ( *(_BYTE *)(v10 - 8) == 78 )
        {
          v11 = *(_QWORD *)(v10 - 48);
          if ( !*(_BYTE *)(v11 + 16) && *(_DWORD *)(v11 + 36) == 79 )
          {
            v12 = *(_QWORD *)(((v10 - 24) & 0xFFFFFFFFFFFFFFF8LL)
                            - 24LL * (*(_DWORD *)(((v10 - 24) & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
            if ( v12 )
            {
              v14 = a5;
              v13 = sub_148B0D0(a1, a3, a4, a5, v12, 0);
              a5 = v14;
              if ( v13 )
                break;
            }
          }
        }
        v10 = *(_QWORD *)(v10 + 8);
      }
      while ( v9 != v10 );
    }
    return v10 != v9;
  }
  return result;
}
