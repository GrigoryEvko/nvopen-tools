// Function: sub_EB5B20
// Address: 0xeb5b20
//
__int64 __fastcall sub_EB5B20(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  int v4; // eax
  __int64 v6; // rbx
  const char *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD v10[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v11; // [rsp+30h] [rbp-30h]

  v3 = *(_QWORD *)(a1 + 328);
  if ( *(_QWORD *)(a1 + 320) != v3 && *(_BYTE *)(v3 - 3) )
  {
    sub_EB4E00(a1);
    return 0;
  }
  else
  {
    if ( a3 )
    {
      v4 = **(_DWORD **)(a1 + 48);
      if ( v4 == 9 )
      {
        v6 = 39;
        v7 = ".error directive invoked in source file";
      }
      else
      {
        if ( v4 != 3 )
        {
          v10[0] = ".error argument must be a string";
          v11 = 259;
          return sub_ECE0E0(a1, v10, 0, 0);
        }
        v8 = sub_ECD7B0(a1);
        v6 = *(_QWORD *)(v8 + 16);
        v7 = *(const char **)(v8 + 8);
        if ( v6 )
        {
          v9 = v6 - 1;
          if ( !v9 )
            v9 = 1;
          ++v7;
          v6 = v9 - 1;
        }
        sub_EABFE0(a1);
      }
      v10[0] = v7;
      v11 = 261;
      v10[1] = v6;
    }
    else
    {
      v10[0] = ".err encountered";
      v11 = 259;
    }
    return sub_ECDA70(a1, a2, v10, 0, 0);
  }
}
