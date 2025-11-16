// Function: sub_24C0860
// Address: 0x24c0860
//
__int64 __fastcall sub_24C0860(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // r12
  const char *v4; // rax
  unsigned __int64 v5; // rdx
  const char *v6; // rax
  unsigned __int64 v7; // rdx
  const char *v8; // rax
  unsigned __int64 v9; // rdx
  const char *v10; // rdi
  unsigned __int64 v11; // rdx
  const char *v12; // rdi
  unsigned __int64 v13; // rdx

  v1 = 0;
  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v2 + 24) )
    return v1;
  if ( (*(_BYTE *)(v2 + 33) & 0x20) == 0 )
  {
    v1 = sub_B2D610(*(_QWORD *)(a1 - 32), 36);
    if ( !(_BYTE)v1 )
    {
      v4 = sub_BD5D20(v2);
      if ( v5 <= 6 || *(_DWORD *)v4 != 1935761247 || *((_WORD *)v4 + 2) != 28257 || v4[6] != 95 )
      {
        v6 = sub_BD5D20(v2);
        if ( v7 <= 7 || *(_QWORD *)v6 != 0x5F6E617377685F5FLL )
        {
          v8 = sub_BD5D20(v2);
          if ( v9 <= 7 || *(_QWORD *)v8 != 0x5F6E617362755F5FLL )
          {
            v10 = sub_BD5D20(v2);
            if ( v11 <= 6 || memcmp(v10, "__msan_", 7u) )
            {
              v12 = sub_BD5D20(v2);
              if ( v13 > 6 )
                LOBYTE(v1) = memcmp(v12, "__tsan_", 7u) == 0;
              return v1;
            }
          }
        }
      }
    }
  }
  return 1;
}
