// Function: sub_18151B0
// Address: 0x18151b0
//
__int64 __fastcall sub_18151B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  const char *v4; // rax
  __int64 v5; // rdx
  int v6; // r9d
  __int64 v7; // r13
  const char *v8; // rax
  __int64 v9; // rdx
  int v10; // r9d
  __int64 v11; // r13
  const char *v12; // rax
  __int64 v13; // rdx
  int v14; // r9d
  char v15; // r8

  if ( (unsigned __int8)sub_3946E40(
                          *(_QWORD *)(a1 + 392),
                          (unsigned int)"dataflow",
                          8,
                          (unsigned int)"src",
                          3,
                          *(_QWORD *)(a1 + 392),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 184LL),
                          (__int64)"functional",
                          10) )
    return 2;
  v3 = *(_QWORD *)(a1 + 392);
  v4 = sub_1649960(a2);
  if ( (unsigned __int8)sub_3946E40(
                          v3,
                          (unsigned int)"dataflow",
                          8,
                          (unsigned int)"fun",
                          3,
                          v6,
                          (__int64)v4,
                          v5,
                          (__int64)"functional",
                          10) )
    return 2;
  if ( (unsigned __int8)sub_3946E40(
                          *(_QWORD *)(a1 + 392),
                          (unsigned int)"dataflow",
                          8,
                          (unsigned int)"src",
                          3,
                          *(_QWORD *)(a1 + 392),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 184LL),
                          (__int64)"discard",
                          7) )
    return 1;
  v7 = *(_QWORD *)(a1 + 392);
  v8 = sub_1649960(a2);
  if ( (unsigned __int8)sub_3946E40(
                          v7,
                          (unsigned int)"dataflow",
                          8,
                          (unsigned int)"fun",
                          3,
                          v10,
                          (__int64)v8,
                          v9,
                          (__int64)"discard",
                          7) )
    return 1;
  if ( (unsigned __int8)sub_3946E40(
                          *(_QWORD *)(a1 + 392),
                          (unsigned int)"dataflow",
                          8,
                          (unsigned int)"src",
                          3,
                          *(_QWORD *)(a1 + 392),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 176LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 184LL),
                          (__int64)"custom",
                          6) )
    return 3;
  v11 = *(_QWORD *)(a1 + 392);
  v12 = sub_1649960(a2);
  v15 = sub_3946E40(
          v11,
          (unsigned int)"dataflow",
          8,
          (unsigned int)"fun",
          3,
          v14,
          (__int64)v12,
          v13,
          (__int64)"custom",
          6);
  result = 0;
  if ( v15 )
    return 3;
  return result;
}
