// Function: sub_318C240
// Address: 0x318c240
//
_QWORD *__fastcall sub_318C240(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *result; // rax
  unsigned __int8 *v11; // rdi
  _QWORD *v12; // [rsp+8h] [rbp-18h]

  result = sub_318C0B0(a1, a2, a4, a5, a5, a6, a7, a8, a9);
  v11 = (unsigned __int8 *)result[2];
  if ( *v11 == 41 )
  {
    v12 = result;
    sub_B45260(v11, *(_QWORD *)(a3 + 16), 1);
    return v12;
  }
  return result;
}
