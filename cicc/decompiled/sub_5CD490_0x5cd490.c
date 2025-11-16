// Function: sub_5CD490
// Address: 0x5cd490
//
__int64 __fastcall sub_5CD490(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  _QWORD *v5; // r15
  const char *v6; // rsi
  const char *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // [rsp+8h] [rbp-68h]
  _QWORD v11[12]; // [rsp+10h] [rbp-60h] BYREF

  v11[0] = "global-dynamic";
  v3 = *(_QWORD *)(a1 + 48);
  v11[1] = "local-dynamic";
  v11[2] = "initial-exec";
  v11[3] = "local-exec";
  v11[4] = 0;
  if ( (*(_BYTE *)(a2 + 140) & 1) != 0 || v3 && (*(_BYTE *)(v3 + 224) & 1) != 0 )
  {
    v5 = v11;
    v6 = "global-dynamic";
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
    v7 = *(const char **)(v10 + 184);
    while ( strcmp(v7, v6) )
    {
      v6 = (const char *)v5[1];
      ++v5;
      if ( !v6 )
      {
        sub_6851C0(1868, a1 + 56);
        *(_BYTE *)(a1 + 8) = 0;
        return a2;
      }
    }
    if ( v3 )
    {
      if ( (*(_BYTE *)(v3 + 127) & 0x10) == 0 )
      {
        v8 = sub_736C60(62, *(_QWORD *)(a2 + 104));
        v9 = v8;
        if ( v8 )
        {
          if ( strcmp(*(const char **)(v10 + 184), *(const char **)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL) + 184LL)) )
          {
            sub_6854F0(8, 1869, *(_QWORD *)(a1 + 32) + 24LL, v9 + 56);
            *(_BYTE *)(a1 + 8) = 0;
          }
        }
      }
    }
  }
  else
  {
    sub_5CCAE0(5u, a1);
  }
  return a2;
}
