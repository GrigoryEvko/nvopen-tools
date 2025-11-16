// Function: sub_204D4D0
// Address: 0x204d4d0
//
char __fastcall sub_204D4D0(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // dl
  unsigned int v5; // eax
  char v6; // dl
  char result; // al
  __int64 v8; // r8
  unsigned int v9; // edx
  char v10; // al
  _QWORD *v11; // rsi
  __int64 v12; // r14
  unsigned int v13; // eax
  _QWORD *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r13
  unsigned int v17; // r15d
  char v18[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h]

  v4 = *(_BYTE *)(a3 + 8);
  if ( v4 == 15 )
  {
    v5 = 8 * sub_15A9520(a2, *(_DWORD *)(a3 + 8) >> 8);
    if ( v5 == 32 )
      return 5;
    if ( v5 > 0x20 )
    {
      v6 = 6;
      if ( v5 != 64 )
      {
        v6 = 0;
        if ( v5 == 128 )
          return 7;
      }
    }
    else
    {
      v6 = 3;
      if ( v5 != 8 )
        return 4 * (v5 == 16);
    }
    return v6;
  }
  else if ( v4 == 16 )
  {
    v8 = *(_QWORD *)(a3 + 24);
    if ( *(_BYTE *)(v8 + 8) == 15 )
    {
      v9 = 8 * sub_15A9520(a2, *(_DWORD *)(v8 + 8) >> 8);
      if ( v9 == 32 )
      {
        v10 = 5;
      }
      else if ( v9 > 0x20 )
      {
        v10 = 6;
        if ( v9 != 64 )
        {
          v10 = 0;
          if ( v9 == 128 )
            v10 = 7;
        }
      }
      else
      {
        v10 = 3;
        if ( v9 != 8 )
          v10 = 4 * (v9 == 16);
      }
      v11 = *(_QWORD **)a3;
      v18[0] = v10;
      v19 = 0;
      v8 = sub_1F58E60((__int64)v18, v11);
    }
    v12 = *(_QWORD *)(a3 + 32);
    LOBYTE(v13) = sub_1F59570(v8);
    v14 = *(_QWORD **)a3;
    v16 = v15;
    v17 = v13;
    result = sub_1D15020(v13, v12);
    if ( !result )
      return sub_1F593D0(v14, v17, v16, v12);
  }
  else
  {
    return sub_1F59570(a3);
  }
  return result;
}
