// Function: sub_228DA60
// Address: 0x228da60
//
bool __fastcall sub_228DA60(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r13
  __int64 v6; // rdx
  __int64 *v7; // r15
  unsigned __int8 v8; // bl
  _QWORD *v9; // rax
  __int64 v10; // rax
  _QWORD **v11; // rsi
  unsigned int v12; // eax
  __int64 v14; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v5 = a3;
    if ( *(_WORD *)(a2 + 24) != 8 )
      return sub_228D760(a1, a2, a3);
    if ( !a3 )
      break;
    v6 = *(_QWORD *)(a2 + 48);
    v7 = (__int64 *)a4;
    v8 = a5;
    v9 = v5;
    while ( (_QWORD *)v6 != v9 )
    {
      v9 = (_QWORD *)*v9;
      if ( !v9 )
        return 0;
    }
    v14 = **(_QWORD **)(a2 + 32);
    v10 = sub_D33D80((_QWORD *)a2, *(_QWORD *)(a1 + 8), v6, a4, a5);
    if ( !sub_228D760(a1, v10, v5) )
      break;
    v11 = *(_QWORD ***)(a2 + 48);
    if ( v8 )
      v12 = sub_228D710(a1, v11);
    else
      v12 = sub_228D730(a1, v11);
    sub_228AC40(v7, v12);
    a2 = v14;
    a5 = v8;
    a4 = (__int64)v7;
    a3 = v5;
  }
  return 0;
}
