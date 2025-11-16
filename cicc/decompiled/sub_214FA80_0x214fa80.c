// Function: sub_214FA80
// Address: 0x214fa80
//
unsigned __int64 __fastcall sub_214FA80(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // [rsp-48h] [rbp-48h]
  _QWORD v11[2]; // [rsp-28h] [rbp-28h] BYREF
  __int16 v12; // [rsp-18h] [rbp-18h]
  __int64 v13; // [rsp-8h] [rbp-8h]

  if ( a2 == 4 )
  {
    v9 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v9;
    if ( result <= 4 )
    {
      return sub_16E7EE0(a3, "const", 5u);
    }
    else
    {
      *(_DWORD *)v9 = 1936617315;
      *(_BYTE *)(v9 + 4) = 116;
      *(_QWORD *)(a3 + 24) += 5LL;
    }
  }
  else if ( a2 > 4 )
  {
    if ( a2 != 5 )
      goto LABEL_18;
    v7 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v7;
    if ( result <= 4 )
    {
      return sub_16E7EE0(a3, "local", 5u);
    }
    else
    {
      *(_DWORD *)v7 = 1633906540;
      *(_BYTE *)(v7 + 4) = 108;
      *(_QWORD *)(a3 + 24) += 5LL;
    }
  }
  else
  {
    if ( a2 != 1 )
    {
      if ( a2 == 3 )
      {
        v5 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v5) <= 5 )
          return sub_16E7EE0(a3, "shared", 6u);
        *(_DWORD *)v5 = 1918986355;
        *(_WORD *)(v5 + 4) = 25701;
        *(_QWORD *)(a3 + 24) += 6LL;
        return 25701;
      }
LABEL_18:
      v13 = v3;
      LODWORD(v10) = a2;
      v11[0] = "Bad address space found while emitting PTX: ";
      v12 = 2307;
      v11[1] = v10;
      sub_16BCFB0((__int64)v11, 1u);
    }
    v8 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v8;
    if ( result <= 5 )
    {
      return sub_16E7EE0(a3, "global", 6u);
    }
    else
    {
      *(_DWORD *)v8 = 1651469415;
      *(_WORD *)(v8 + 4) = 27745;
      *(_QWORD *)(a3 + 24) += 6LL;
    }
  }
  return result;
}
