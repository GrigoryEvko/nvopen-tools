// Function: sub_240DD00
// Address: 0x240dd00
//
__int64 __fastcall sub_240DD00(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // eax
  char *v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // r9
  char *v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // r12
  unsigned __int64 v14; // rcx
  char *v15; // rsi
  int v16; // eax
  unsigned __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-38h]

  v2 = 0;
  LOBYTE(v4) = sub_23C76F0(
                 *(_QWORD *)(a1 + 792),
                 "dataflow",
                 8u,
                 "src",
                 3u,
                 *(_QWORD *)(a1 + 792),
                 *(char **)(*(_QWORD *)(a2 + 40) + 168LL),
                 *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
                 "uninstrumented",
                 0xEu);
  if ( !(_BYTE)v4 )
  {
    v2 = v4;
    v18 = *(_QWORD *)(a1 + 792);
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 24) + 8LL) == 13 )
    {
      v9 = (char *)sub_BD5D20(a2);
      LOBYTE(v11) = sub_23C76F0(v18, "dataflow", 8u, "fun", 3u, v18, v9, v10, "uninstrumented", 0xEu);
      return v11 ^ 1u;
    }
    else
    {
      v5 = (char *)sub_BD5D20(a2);
      if ( !sub_23C76F0(v18, "dataflow", 8u, "global", 6u, v18, v5, v6, "uninstrumented", 0xEu) )
      {
        v12 = *(_QWORD *)(a2 + 24);
        v13 = *(_QWORD *)(a1 + 792);
        v14 = 14;
        v15 = "<unknown type>";
        if ( *(_BYTE *)(v12 + 8) == 15 && (*(_BYTE *)(v12 + 9) & 4) == 0 )
        {
          v15 = (char *)sub_BCB490(v12);
          v14 = v17;
        }
        LOBYTE(v16) = sub_23C76F0(v13, "dataflow", 8u, "type", 4u, v7, v15, v14, "uninstrumented", 0xEu);
        return v16 ^ 1u;
      }
    }
  }
  return v2;
}
