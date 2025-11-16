// Function: sub_1814820
// Address: 0x1814820
//
__int64 __fastcall sub_1814820(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // eax
  const char *v5; // rax
  __int64 v6; // rdx
  int v7; // r9d
  const char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // rcx
  const char *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v4 = sub_3946E40(
         *(_QWORD *)(a1 + 392),
         (unsigned int)"dataflow",
         8,
         (unsigned int)"src",
         3,
         *(_QWORD *)(a1 + 392),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 184LL),
         (__int64)"uninstrumented",
         14);
  if ( !(_BYTE)v4 )
  {
    v2 = v4;
    v16 = *(_QWORD *)(a1 + 392);
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 24) + 8LL) == 12 )
    {
      v9 = sub_1649960(a2);
      return (unsigned int)sub_3946E40(
                             v16,
                             (unsigned int)"dataflow",
                             8,
                             (unsigned int)"fun",
                             3,
                             v16,
                             (__int64)v9,
                             v10,
                             (__int64)"uninstrumented",
                             14)
           ^ 1;
    }
    else
    {
      v5 = sub_1649960(a2);
      if ( !(unsigned __int8)sub_3946E40(
                               v16,
                               (unsigned int)"dataflow",
                               8,
                               (unsigned int)"global",
                               6,
                               v16,
                               (__int64)v5,
                               v6,
                               (__int64)"uninstrumented",
                               14) )
      {
        v11 = *(_QWORD *)(a2 + 24);
        v12 = *(_QWORD *)(a1 + 392);
        v13 = 14;
        v14 = "<unknown type>";
        if ( *(_BYTE *)(v11 + 8) == 13 && (*(_BYTE *)(v11 + 9) & 4) == 0 )
        {
          v14 = (const char *)sub_1643640(v11);
          v13 = v15;
        }
        return (unsigned int)sub_3946E40(
                               v12,
                               (unsigned int)"dataflow",
                               8,
                               (unsigned int)"type",
                               4,
                               v7,
                               (__int64)v14,
                               v13,
                               (__int64)"uninstrumented",
                               14)
             ^ 1;
      }
    }
  }
  return v2;
}
