// Function: sub_390C370
// Address: 0x390c370
//
__int64 __fastcall sub_390C370(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned __int8 v6; // al
  unsigned int v7; // r8d
  unsigned __int8 v9; // [rsp+Fh] [rbp-49h] BYREF
  __int64 v10; // [rsp+10h] [rbp-48h] BYREF
  _QWORD v11[3]; // [rsp+18h] [rbp-40h] BYREF
  int v12; // [rsp+30h] [rbp-28h]

  memset(v11, 0, sizeof(v11));
  v12 = 0;
  v6 = sub_390B240(a1, a4, a2, a3, v11, &v10, &v9);
  if ( !v11[0] )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 8) + 88LL))(
             *(_QWORD *)(a1 + 8),
             a2,
             v6,
             v10,
             a3,
             a4,
             v9);
  if ( *(_WORD *)(v11[0] + 16LL) != 28 )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 8) + 88LL))(
             *(_QWORD *)(a1 + 8),
             a2,
             v6,
             v10,
             a3,
             a4,
             v9);
  v7 = 0;
  if ( *(_DWORD *)(a2 + 12) )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 8) + 88LL))(
             *(_QWORD *)(a1 + 8),
             a2,
             v6,
             v10,
             a3,
             a4,
             v9);
  return v7;
}
