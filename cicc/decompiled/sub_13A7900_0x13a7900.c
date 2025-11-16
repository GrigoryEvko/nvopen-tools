// Function: sub_13A7900
// Address: 0x13a7900
//
__int64 __fastcall sub_13A7900(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // r15
  __int64 v18; // rax

  v4 = sub_1456040(a2);
  if ( *(_BYTE *)(v4 + 8) != 11 )
  {
    sub_1456040(a3);
    return 0;
  }
  v5 = v4;
  v6 = sub_1456040(a3);
  if ( *(_BYTE *)(v6 + 8) != 11 )
    return 0;
  if ( *(_DWORD *)(v5 + 8) >> 8 < *(_DWORD *)(v6 + 8) >> 8 )
    v5 = v6;
  v7 = sub_1483B20(*(_QWORD *)(a1 + 8), a2, v5);
  v8 = sub_1483B20(*(_QWORD *)(a1 + 8), a3, v5);
  v9 = sub_14806B0(*(_QWORD *)(a1 + 8), v7, v8, 0, 0);
  v10 = v9;
  if ( *(_WORD *)(v9 + 24) != 7
    || *(_QWORD *)(v9 + 40) != 2
    || (v17 = sub_1481F60(*(_QWORD *)(a1 + 8), *(_QWORD *)(v9 + 48)), (unsigned __int8)sub_14562D0(v17))
    || (v18 = sub_1487810(v10, v17, *(_QWORD *)(a1 + 8)), result = sub_1477B50(*(_QWORD *)(a1 + 8), v18), !(_BYTE)result) )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v12 = sub_1456040(v8);
    v13 = sub_145CF80(v11, v12, 1, 0);
    v14 = sub_147A9C0(v11, v8, v13);
    v15 = sub_14806B0(v11, v7, v14, 0, 0);
    return sub_1477B50(*(_QWORD *)(a1 + 8), v15);
  }
  return result;
}
