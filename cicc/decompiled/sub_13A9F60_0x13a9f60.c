// Function: sub_13A9F60
// Address: 0x13a9f60
//
__int64 __fastcall sub_13A9F60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // r14d
  __int64 v17; // r9
  int v19; // eax
  int v20; // r14d
  __int64 v21; // r9
  unsigned int v22; // [rsp+Ch] [rbp-54h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27[8]; // [rsp+20h] [rbp-40h] BYREF

  v22 = a6 - 1;
  *(_BYTE *)(a7 + 43) = 0;
  v10 = sub_14806B0(*(_QWORD *)(a1 + 8), a3, a4, 0, 0);
  v24 = *(_QWORD *)(a1 + 8);
  v11 = sub_1456040(v10);
  v12 = sub_145CF80(v24, v11, 0, 0);
  sub_13A62E0(a8, v12, a2, v10, a5);
  if ( (unsigned __int8)sub_13A7760(a1, 32, a3, a4) )
  {
    if ( *(_DWORD *)(a1 + 32) > v22 )
    {
      v17 = 16LL * v22;
      *(_BYTE *)(v17 + *(_QWORD *)(a7 + 48)) &= ~1u;
      *(_BYTE *)(*(_QWORD *)(a7 + 48) + v17) |= 0x10u;
    }
    return 0;
  }
  if ( *(_WORD *)(a2 + 24) )
    return 0;
  v25 = a2;
  if ( (unsigned __int8)sub_1477B50(*(_QWORD *)(a1 + 8), a2) )
    v25 = sub_1480620(*(_QWORD *)(a1 + 8), a2, 0);
  v13 = v10;
  if ( (unsigned __int8)sub_1477B50(*(_QWORD *)(a1 + 8), a2) )
    v13 = sub_1480620(*(_QWORD *)(a1 + 8), v10, 0);
  v14 = sub_1456040(v10);
  v15 = sub_13A7AF0(a1, a5, v14);
  if ( !v15 )
  {
LABEL_14:
    if ( (unsigned __int8)sub_1477B50(*(_QWORD *)(a1 + 8), v13) )
      return 1;
    if ( !*(_WORD *)(v10 + 24) )
    {
      sub_16AB4D0(v27, *(_QWORD *)(v10 + 32) + 24LL, *(_QWORD *)(a2 + 32) + 24LL);
      LOBYTE(v19) = sub_13A38F0((__int64)v27, 0);
      v20 = v19;
      sub_135E100(v27);
      return v20 ^ 1u;
    }
    return 0;
  }
  v26 = sub_13A5B60(*(_QWORD *)(a1 + 8), v25, v15, 0, 0);
  v16 = sub_13A7760(a1, 38, v13, v26);
  if ( !(_BYTE)v16 )
  {
    if ( (unsigned __int8)sub_13A7760(a1, 32, v13, v26) )
    {
      if ( *(_DWORD *)(a1 + 32) > v22 )
      {
        v21 = 16LL * v22;
        *(_BYTE *)(v21 + *(_QWORD *)(a7 + 48)) &= ~4u;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + v21) |= 0x20u;
        return v16;
      }
      return 0;
    }
    goto LABEL_14;
  }
  return 1;
}
