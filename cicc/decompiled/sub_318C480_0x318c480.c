// Function: sub_318C480
// Address: 0x318c480
//
_QWORD *__fastcall sub_318C480(
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
  _QWORD *v10; // rax
  _QWORD *v11; // r12

  v10 = sub_318C2A0(a1, a2, a3, a5, a6, a6, a7, a8, a9);
  v11 = v10;
  if ( *((_DWORD *)v10 + 2) == 51 )
    sub_B45260((unsigned __int8 *)v10[2], *(_QWORD *)(a4 + 16), 1);
  return v11;
}
