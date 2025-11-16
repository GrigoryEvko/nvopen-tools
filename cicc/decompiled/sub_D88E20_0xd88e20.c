// Function: sub_D88E20
// Address: 0xd88e20
//
__int64 __fastcall sub_D88E20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int v8; // r15d
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13

  if ( a5 )
    return 0;
  v8 = *(_DWORD *)(a1 + 24);
  v9 = (_QWORD *)sub_B2BE50(**(_QWORD **)(a1 + 16));
  v10 = sub_BCD140(v9, v8);
  v11 = sub_DA2C50(*(_QWORD *)(a1 + 16), v10, a4, 0);
  v12 = v11;
  if ( !a3 )
    return 1;
  if ( (unsigned __int8)sub_D96A50(v11) )
    return 0;
  return sub_D88C60(a1, *a2, a2[3], a3, v12);
}
