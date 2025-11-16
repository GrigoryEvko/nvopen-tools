// Function: sub_171D0D0
// Address: 0x171d0d0
//
unsigned __int8 *__fastcall sub_171D0D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        char a5,
        char a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // r12
  __int64 v10; // rax

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
    return sub_170A2B0(a1, 13, (__int64 *)a2, a3, a4, a5, a6);
  v9 = sub_15A2B60((__int64 *)a2, a3, a5, a6, a7, a8, a9);
  v10 = sub_14DBA30(v9, *(_QWORD *)(a1 + 96), 0);
  if ( v10 )
    return (unsigned __int8 *)v10;
  return (unsigned __int8 *)v9;
}
