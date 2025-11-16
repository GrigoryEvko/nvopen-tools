// Function: sub_240F1D0
// Address: 0x240f1d0
//
__int64 __fastcall sub_240F1D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  char *v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // r9
  __int64 v7; // r13
  char *v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r13
  char *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r9
  bool v15; // r8

  if ( sub_23C76F0(
         *(_QWORD *)(a1 + 792),
         "dataflow",
         8u,
         "src",
         3u,
         *(_QWORD *)(a1 + 792),
         *(char **)(*(_QWORD *)(a2 + 40) + 168LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
         "functional",
         0xAu) )
  {
    return 2;
  }
  v3 = *(_QWORD *)(a1 + 792);
  v4 = (char *)sub_BD5D20(a2);
  if ( sub_23C76F0(v3, "dataflow", 8u, "fun", 3u, v6, v4, v5, "functional", 0xAu) )
    return 2;
  if ( sub_23C76F0(
         *(_QWORD *)(a1 + 792),
         "dataflow",
         8u,
         "src",
         3u,
         *(_QWORD *)(a1 + 792),
         *(char **)(*(_QWORD *)(a2 + 40) + 168LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
         "discard",
         7u) )
  {
    return 1;
  }
  v7 = *(_QWORD *)(a1 + 792);
  v8 = (char *)sub_BD5D20(a2);
  if ( sub_23C76F0(v7, "dataflow", 8u, "fun", 3u, v10, v8, v9, "discard", 7u) )
    return 1;
  if ( sub_23C76F0(
         *(_QWORD *)(a1 + 792),
         "dataflow",
         8u,
         "src",
         3u,
         *(_QWORD *)(a1 + 792),
         *(char **)(*(_QWORD *)(a2 + 40) + 168LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
         "custom",
         6u) )
  {
    return 3;
  }
  v11 = *(_QWORD *)(a1 + 792);
  v12 = (char *)sub_BD5D20(a2);
  v15 = sub_23C76F0(v11, "dataflow", 8u, "fun", 3u, v14, v12, v13, "custom", 6u);
  result = 0;
  if ( v15 )
    return 3;
  return result;
}
