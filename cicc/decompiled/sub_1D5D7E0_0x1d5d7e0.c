// Function: sub_1D5D7E0
// Address: 0x1d5d7e0
//
__int64 __fastcall sub_1D5D7E0(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  __int64 v3; // rbx
  char v5; // cl
  unsigned int v6; // eax
  unsigned __int8 v7; // dl
  __int64 v9; // r8
  unsigned int v10; // edx
  char v11; // al
  __int64 v12; // rsi
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // r13
  unsigned int v18; // r15d
  __int64 v19; // rax
  char v20[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h]

  v5 = *((_BYTE *)a2 + 8);
  if ( v5 == 15 )
  {
    v6 = 8 * sub_15A9520(a1, *((_DWORD *)a2 + 2) >> 8);
    if ( v6 == 32 )
    {
      return 5;
    }
    else if ( v6 > 0x20 )
    {
      v7 = 6;
      if ( v6 != 64 )
      {
        v7 = 0;
        if ( v6 == 128 )
          return 7;
      }
    }
    else
    {
      v7 = 3;
      if ( v6 != 8 )
        return (unsigned __int8)(4 * (v6 == 16));
    }
    return v7;
  }
  else if ( v5 == 16 )
  {
    v9 = a2[3];
    if ( *(_BYTE *)(v9 + 8) == 15 )
    {
      v10 = 8 * sub_15A9520(a1, *(_DWORD *)(v9 + 8) >> 8);
      if ( v10 == 32 )
      {
        v11 = 5;
      }
      else if ( v10 > 0x20 )
      {
        v11 = 6;
        if ( v10 != 64 )
        {
          v11 = 0;
          if ( v10 == 128 )
            v11 = 7;
        }
      }
      else
      {
        v11 = 3;
        if ( v10 != 8 )
          v11 = 4 * (v10 == 16);
      }
      v12 = *a2;
      v20[0] = v11;
      v21 = 0;
      v9 = sub_1F58E60(v20, v12);
    }
    v13 = a2[4];
    v14 = sub_1F59570(v9, 0);
    v15 = *a2;
    v17 = v16;
    v18 = v14;
    LOBYTE(v19) = sub_1D15020(v14, v13);
    if ( !(_BYTE)v19 )
    {
      v19 = sub_1F593D0(v15, v18, v17, (unsigned int)v13);
      v3 = v19;
    }
    LOBYTE(v3) = v19;
    return v3;
  }
  else
  {
    return sub_1F59570(a2, a3);
  }
}
