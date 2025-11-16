// Function: sub_DDC2C0
// Address: 0xddc2c0
//
char __fastcall sub_DDC2C0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, _BYTE *a5)
{
  char result; // al
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r8
  char v13; // al
  _BYTE *v14; // [rsp-40h] [rbp-40h]

  result = *(_BYTE *)(a1 + 16);
  if ( result )
  {
    v9 = a2 + 48;
    v10 = *(_QWORD *)(a2 + 56);
    if ( a2 + 48 != v10 )
    {
      do
      {
        if ( !v10 )
          BUG();
        if ( *(_BYTE *)(v10 - 24) == 85 )
        {
          v11 = *(_QWORD *)(v10 - 56);
          if ( v11 )
          {
            if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v10 + 56) && *(_DWORD *)(v11 + 36) == 153 )
            {
              v12 = *(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 - 20) & 0x7FFFFFF) - 24);
              if ( v12 )
              {
                v14 = a5;
                v13 = sub_DDBDC0(a1, a3, a4, a5, v12, 0, 0);
                a5 = v14;
                if ( v13 )
                  break;
              }
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
