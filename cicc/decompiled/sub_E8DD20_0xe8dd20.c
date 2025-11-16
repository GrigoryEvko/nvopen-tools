// Function: sub_E8DD20
// Address: 0xe8dd20
//
__int64 *__fastcall sub_E8DD20(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 *(__fastcall *v8)(__int64 *, __int64, __int64); // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9

  v5 = sub_E6C430(a1[1], (__int64)a2, a3, a4, a5);
  *a2 = v5;
  v6 = v5;
  v7 = v5;
  v8 = *(__int64 *(__fastcall **)(__int64 *, __int64, __int64))(*a1 + 208);
  if ( v8 != sub_E8DC70 )
    return v8(a1, v7, 0);
  sub_E98820(a1, v7, 0);
  sub_E5CB20(a1[37], v6, v9, v10, v11, v12);
  v13 = sub_E8BB10(a1, 0);
  *(_QWORD *)v6 = v13;
  *(_QWORD *)(v6 + 24) = *(_QWORD *)(v13 + 48);
  *(_BYTE *)(v6 + 9) = *(_BYTE *)(v6 + 9) & 0x8F | 0x10;
  return sub_E8DAF0((__int64)a1, v6, v14, v15, v16, v17);
}
