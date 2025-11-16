// Function: sub_39B25A0
// Address: 0x39b25a0
//
__int64 __fastcall sub_39B25A0(__int64 a1, __int64 a2)
{
  char v4; // al
  __int64 v5; // rdi
  unsigned int v6; // edx
  unsigned int v7; // r8d
  unsigned __int8 v8; // al
  __int64 v10; // r8
  unsigned int v11; // edx
  char v12; // al
  _QWORD *v13; // rsi
  __int64 v14; // r14
  unsigned int v15; // eax
  _QWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  unsigned int v19; // r15d
  char v20[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h]

  v4 = *(_BYTE *)(a2 + 8);
  v5 = *(_QWORD *)(a1 + 8);
  if ( v4 == 15 )
  {
    v6 = 8 * sub_15A9520(v5, *(_DWORD *)(a2 + 8) >> 8);
    if ( v6 == 32 )
    {
      v8 = 5;
    }
    else if ( v6 > 0x20 )
    {
      if ( v6 == 64 )
      {
        v8 = 6;
      }
      else
      {
        if ( v6 != 128 )
          return 0;
        v8 = 7;
      }
    }
    else if ( v6 == 8 )
    {
      v8 = 3;
    }
    else
    {
      v8 = 4;
      if ( v6 != 16 )
        return 0;
    }
    goto LABEL_11;
  }
  if ( v4 != 16 )
  {
    v8 = sub_1F59570(a2);
    v7 = 0;
    if ( !v8 )
      return v7;
    goto LABEL_11;
  }
  v10 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)(v10 + 8) == 15 )
  {
    v11 = 8 * sub_15A9520(v5, *(_DWORD *)(v10 + 8) >> 8);
    if ( v11 == 32 )
    {
      v12 = 5;
    }
    else if ( v11 > 0x20 )
    {
      v12 = 6;
      if ( v11 != 64 )
      {
        v12 = 0;
        if ( v11 == 128 )
          v12 = 7;
      }
    }
    else
    {
      v12 = 3;
      if ( v11 != 8 )
        v12 = 4 * (v11 == 16);
    }
    v13 = *(_QWORD **)a2;
    v20[0] = v12;
    v21 = 0;
    v10 = sub_1F58E60((__int64)v20, v13);
  }
  v14 = *(_QWORD *)(a2 + 32);
  LOBYTE(v15) = sub_1F59570(v10);
  v16 = *(_QWORD **)a2;
  v18 = v17;
  v19 = v15;
  v8 = sub_1D15020(v15, v14);
  if ( !v8 )
    v8 = sub_1F593D0(v16, v19, v18, v14);
  v7 = 0;
  if ( v8 )
LABEL_11:
    LOBYTE(v7) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v8 + 120) != 0;
  return v7;
}
