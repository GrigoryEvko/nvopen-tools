// Function: sub_1B97BE0
// Address: 0x1b97be0
//
char __fastcall sub_1B97BE0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // r14
  char v3; // al
  char result; // al
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // r15
  __int64 v11; // r14
  char v12; // r8
  int v13; // r15d
  unsigned int v14; // r13d
  unsigned int v15; // [rsp+4h] [rbp-3Ch] BYREF
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = **(_QWORD **)a1;
  v3 = *(_BYTE *)(v2 + 16);
  if ( v3 == 77 )
    return (unsigned __int8)(v3 - 54) > 1u;
  v5 = *a2;
  v6 = *(_QWORD *)(a1 + 8);
  v15 = v5;
  v8 = *(_QWORD *)(v6 + 32);
  if ( v5 == 1 )
    return 0;
  v9 = (unsigned __int8)sub_1B97860(v8 + 200, (int *)&v15, v16)
     ? v16[0]
     : *(_QWORD *)(v8 + 208) + 80LL * *(unsigned int *)(v8 + 224);
  if ( sub_13A0E30(v9 + 8, v2) )
    return 0;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = **(_QWORD **)a1;
  v12 = sub_1B918B0(*(_QWORD *)(v10 + 32), v11, v5);
  result = 0;
  if ( !v12 )
  {
    v3 = *(_BYTE *)(v11 + 16);
    if ( v3 != 78 )
      return (unsigned __int8)(v3 - 54) > 1u;
    v13 = sub_14C3B40(v11, *(__int64 **)(v10 + 8));
    v14 = sub_1B8FD60(v11, v5, *(__int64 **)(*(_QWORD *)(a1 + 8) + 16LL), *(__int64 **)(*(_QWORD *)(a1 + 8) + 8LL), v16);
    if ( v13
      && v14 >= (unsigned int)sub_1B8F4A0(
                                v11,
                                v5,
                                *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL),
                                *(__int64 **)(*(_QWORD *)(a1 + 8) + 8LL)) )
    {
      return 1;
    }
    else
    {
      return LOBYTE(v16[0]) ^ 1;
    }
  }
  return result;
}
