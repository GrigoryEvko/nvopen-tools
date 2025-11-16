// Function: sub_228E170
// Address: 0x228e170
//
__int64 __fastcall sub_228E170(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // r13
  _QWORD *v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // r14
  __int64 *v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 result; // rax
  __int64 v18; // r15
  _QWORD *v19; // rax

  v4 = sub_D95540(a2);
  if ( *(_BYTE *)(v4 + 8) != 12 )
  {
    sub_D95540(a3);
    return 0;
  }
  v5 = v4;
  v6 = sub_D95540(a3);
  if ( *(_BYTE *)(v6 + 8) != 12 )
    return 0;
  if ( *(_DWORD *)(v5 + 8) >> 8 < *(_DWORD *)(v6 + 8) >> 8 )
    v5 = v6;
  v7 = sub_DC5760(*(_QWORD *)(a1 + 8), a2, v5, 0);
  v8 = sub_DC5760(*(_QWORD *)(a1 + 8), a3, v5, 0);
  v9 = sub_DCC810(*(__int64 **)(a1 + 8), (__int64)v7, (__int64)v8, 0, 0);
  v10 = (__int64)v9;
  if ( *((_WORD *)v9 + 12) != 8
    || v9[5] != 2
    || (v18 = sub_DCF3A0(*(__int64 **)(a1 + 8), (char *)v9[6], 0), sub_D96A50(v18))
    || (v19 = sub_DD0540(v10, v18, *(__int64 **)(a1 + 8)),
        result = sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v19),
        !(_BYTE)result) )
  {
    v11 = *(__int64 **)(a1 + 8);
    v12 = sub_D95540((__int64)v8);
    v13 = sub_DA2C50((__int64)v11, v12, 1, 0);
    v15 = sub_DCDFA0(v11, (__int64)v8, (__int64)v13, v14);
    v16 = sub_DCC810(v11, (__int64)v7, v15, 0, 0);
    return sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v16);
  }
  return result;
}
