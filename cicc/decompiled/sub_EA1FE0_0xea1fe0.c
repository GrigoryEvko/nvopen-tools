// Function: sub_EA1FE0
// Address: 0xea1fe0
//
__int64 __fastcall sub_EA1FE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char *a6)
{
  unsigned int v6; // r13d
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9

  v6 = a3;
  v8 = *(_QWORD *)(a2 + 152);
  v9 = a1[37];
  if ( v8 )
    sub_E5CB20(a1[37], v8, a3, a4, a5, (__int64)a6);
  sub_E8D250(a1, a2, v6, a4, a5, a6);
  return sub_E5CB20(v9, *(_QWORD *)(a2 + 16), v10, v11, v12, v13);
}
