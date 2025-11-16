// Function: sub_39B1E70
// Address: 0x39b1e70
//
char __fastcall sub_39B1E70(__int64 a1, __int64 a2)
{
  char v3; // cl
  unsigned int v4; // eax
  char v5; // dl
  char result; // al
  __int64 v7; // r8
  unsigned int v8; // edx
  char v9; // al
  _QWORD *v10; // rsi
  __int64 v11; // r14
  unsigned int v12; // eax
  _QWORD *v13; // r12
  __int64 v14; // rdx
  __int64 v15; // r13
  unsigned int v16; // r15d
  char v17[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h]

  v3 = *(_BYTE *)(a2 + 8);
  if ( v3 == 15 )
  {
    v4 = 8 * sub_15A9520(a1, *(_DWORD *)(a2 + 8) >> 8);
    if ( v4 == 32 )
      return 5;
    if ( v4 > 0x20 )
    {
      v5 = 6;
      if ( v4 != 64 )
      {
        v5 = 0;
        if ( v4 == 128 )
          return 7;
      }
    }
    else
    {
      v5 = 3;
      if ( v4 != 8 )
        return 4 * (v4 == 16);
    }
    return v5;
  }
  else if ( v3 == 16 )
  {
    v7 = *(_QWORD *)(a2 + 24);
    if ( *(_BYTE *)(v7 + 8) == 15 )
    {
      v8 = 8 * sub_15A9520(a1, *(_DWORD *)(v7 + 8) >> 8);
      if ( v8 == 32 )
      {
        v9 = 5;
      }
      else if ( v8 > 0x20 )
      {
        v9 = 6;
        if ( v8 != 64 )
        {
          v9 = 0;
          if ( v8 == 128 )
            v9 = 7;
        }
      }
      else
      {
        v9 = 3;
        if ( v8 != 8 )
          v9 = 4 * (v8 == 16);
      }
      v10 = *(_QWORD **)a2;
      v17[0] = v9;
      v18 = 0;
      v7 = sub_1F58E60((__int64)v17, v10);
    }
    v11 = *(_QWORD *)(a2 + 32);
    LOBYTE(v12) = sub_1F59570(v7);
    v13 = *(_QWORD **)a2;
    v15 = v14;
    v16 = v12;
    result = sub_1D15020(v12, v11);
    if ( !result )
      return sub_1F593D0(v13, v16, v15, v11);
  }
  else
  {
    return sub_1F59570(a2);
  }
  return result;
}
