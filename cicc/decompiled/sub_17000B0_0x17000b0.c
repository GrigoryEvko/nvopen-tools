// Function: sub_17000B0
// Address: 0x17000b0
//
__int64 __fastcall sub_17000B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax
  __int64 v4; // rdx
  _DWORD *v5; // rcx
  bool v6; // al
  __int64 v7; // rdx
  _DWORD *v8; // rcx
  bool v9; // al
  __int64 v10; // rdx
  _DWORD *v11; // rcx
  bool v12; // al
  __int64 v13; // rdx
  _DWORD *v14; // rcx
  bool v15; // al
  __int64 v16; // rdx
  _DWORD *v17; // rcx
  bool v18; // al
  __int64 v19; // rdx
  __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (_QWORD *)(a2 + 112);
  if ( sub_15602E0((_QWORD *)(a2 + 112), "unsafe-fp-math", 0xEu) )
  {
    v20[0] = sub_1560340(v2, -1, "unsafe-fp-math", 0xEu);
    v17 = (_DWORD *)sub_155D8B0(v20);
    v18 = 0;
    if ( v19 == 4 )
      v18 = *v17 == 1702195828;
    *(_BYTE *)(a1 + 792) = (2 * v18) | *(_BYTE *)(a1 + 792) & 0xFD;
  }
  else
  {
    *(_BYTE *)(a1 + 792) = *(_BYTE *)(a1 + 648) & 2 | *(_BYTE *)(a1 + 792) & 0xFD;
  }
  if ( sub_15602E0(v2, "no-infs-fp-math", 0xFu) )
  {
    v20[0] = sub_1560340(v2, -1, "no-infs-fp-math", 0xFu);
    v14 = (_DWORD *)sub_155D8B0(v20);
    v15 = 0;
    if ( v16 == 4 )
      v15 = *v14 == 1702195828;
    *(_BYTE *)(a1 + 792) = (4 * v15) | *(_BYTE *)(a1 + 792) & 0xFB;
  }
  else
  {
    *(_BYTE *)(a1 + 792) = *(_BYTE *)(a1 + 648) & 4 | *(_BYTE *)(a1 + 792) & 0xFB;
  }
  if ( sub_15602E0(v2, "no-nans-fp-math", 0xFu) )
  {
    v20[0] = sub_1560340(v2, -1, "no-nans-fp-math", 0xFu);
    v11 = (_DWORD *)sub_155D8B0(v20);
    v12 = 0;
    if ( v13 == 4 )
      v12 = *v11 == 1702195828;
    *(_BYTE *)(a1 + 792) = (8 * v12) | *(_BYTE *)(a1 + 792) & 0xF7;
  }
  else
  {
    *(_BYTE *)(a1 + 792) = *(_BYTE *)(a1 + 648) & 8 | *(_BYTE *)(a1 + 792) & 0xF7;
  }
  if ( sub_15602E0(v2, "no-signed-zeros-fp-math", 0x17u) )
  {
    v20[0] = sub_1560340(v2, -1, "no-signed-zeros-fp-math", 0x17u);
    v8 = (_DWORD *)sub_155D8B0(v20);
    v9 = 0;
    if ( v10 == 4 )
      v9 = *v8 == 1702195828;
    *(_BYTE *)(a1 + 792) = (32 * v9) | *(_BYTE *)(a1 + 792) & 0xDF;
  }
  else
  {
    *(_BYTE *)(a1 + 792) = *(_BYTE *)(a1 + 648) & 0x20 | *(_BYTE *)(a1 + 792) & 0xDF;
  }
  if ( sub_15602E0(v2, "no-trapping-math", 0x10u) )
  {
    v20[0] = sub_1560340(v2, -1, "no-trapping-math", 0x10u);
    v5 = (_DWORD *)sub_155D8B0(v20);
    v6 = 0;
    if ( v7 == 4 )
      v6 = *v5 == 1702195828;
    *(_BYTE *)(a1 + 792) = (16 * v6) | *(_BYTE *)(a1 + 792) & 0xEF;
  }
  else
  {
    *(_BYTE *)(a1 + 792) = *(_BYTE *)(a1 + 648) & 0x10 | *(_BYTE *)(a1 + 792) & 0xEF;
  }
  v20[0] = sub_1560340(v2, -1, "denormal-fp-math", 0x10u);
  result = sub_155D8B0(v20);
  if ( v4 == 4 )
  {
    if ( *(_DWORD *)result != 1701143913 )
      goto LABEL_13;
    *(_DWORD *)(a1 + 832) = 0;
  }
  else
  {
    if ( v4 != 13 )
    {
LABEL_13:
      result = *(unsigned int *)(a1 + 688);
      *(_DWORD *)(a1 + 832) = result;
      return result;
    }
    if ( *(_QWORD *)result == 0x6576726573657270LL
      && *(_DWORD *)(result + 8) == 1734964013
      && *(_BYTE *)(result + 12) == 110 )
    {
      *(_DWORD *)(a1 + 832) = 1;
    }
    else
    {
      if ( *(_QWORD *)result != 0x6576697469736F70LL
        || *(_DWORD *)(result + 8) != 1919253037
        || *(_BYTE *)(result + 12) != 111 )
      {
        goto LABEL_13;
      }
      *(_DWORD *)(a1 + 832) = 2;
    }
  }
  return result;
}
