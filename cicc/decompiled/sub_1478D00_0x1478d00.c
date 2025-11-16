// Function: sub_1478D00
// Address: 0x1478d00
//
char __fastcall sub_1478D00(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  char result; // al
  _QWORD *v13; // rax
  __int64 v14; // [rsp-40h] [rbp-40h]

  if ( a2 == (_QWORD *)a3 )
    return 1;
  v4 = **(_QWORD **)(a3 + 32);
  v5 = (__int64 *)a2[4];
  if ( *v5 == v4
    || (v14 = *v5,
        v6 = sub_145DE40(*(_QWORD *)(a1 + 112), *v5, **(_QWORD **)(a3 + 32)),
        sub_1454560(a1 + 128, (__int64)v6))
    || (v13 = sub_145DE40(*(_QWORD *)(a1 + 112), v4, v14), (result = sub_1454560(a1 + 128, (__int64)v13)) != 0) )
  {
    v7 = sub_13A5BC0((_QWORD *)a3, *(_QWORD *)(a1 + 112));
    v8 = sub_13A5BC0(a2, *(_QWORD *)(a1 + 112));
    v9 = v8;
    if ( v7 == v8 )
      return 1;
    v10 = sub_145DE40(*(_QWORD *)(a1 + 112), v8, v7);
    if ( sub_1454560(a1 + 128, (__int64)v10) )
    {
      return 1;
    }
    else
    {
      v11 = sub_145DE40(*(_QWORD *)(a1 + 112), v7, v9);
      return sub_1454560(a1 + 128, (__int64)v11);
    }
  }
  return result;
}
