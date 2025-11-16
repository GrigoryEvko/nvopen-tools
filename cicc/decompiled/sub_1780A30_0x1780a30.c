// Function: sub_1780A30
// Address: 0x1780a30
//
_QWORD *__fastcall sub_1780A30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        double a6,
        double a7,
        double a8)
{
  _QWORD *result; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
    return sub_177F170(a1, (__int64 *)a2, a3, a4, a5);
  v12 = sub_15A2A30((__int64 *)0x10, (__int64 *)a2, a3, 0, 0, a6, a7, a8);
  result = (_QWORD *)sub_14DBA30(v12, *(_QWORD *)(a1 + 96), 0);
  if ( result )
    return result;
  if ( v12 )
    return (_QWORD *)v12;
  else
    return sub_177F170(a1, (__int64 *)a2, a3, a4, a5);
}
