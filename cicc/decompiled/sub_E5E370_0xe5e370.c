// Function: sub_E5E370
// Address: 0xe5e370
//
__int64 __fastcall sub_E5E370(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  unsigned __int8 v5; // al
  unsigned int v6; // r8d
  unsigned __int8 v8; // [rsp+Fh] [rbp-49h] BYREF
  __int64 v9; // [rsp+10h] [rbp-48h] BYREF
  _QWORD v10[3]; // [rsp+18h] [rbp-40h] BYREF
  int v11; // [rsp+30h] [rbp-28h]

  memset(v10, 0, sizeof(v10));
  v4 = *(_QWORD *)(a3 + 32);
  v11 = 0;
  v5 = sub_E5C4E0(a1, (__int64 *)a2, a3, v10, v4, &v9, &v8);
  if ( !v10[0] )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 128LL))(
             *(_QWORD *)(a1 + 8),
             a1,
             a2,
             v5,
             v9,
             a3,
             v8);
  if ( *(_WORD *)(v10[0] + 1LL) != 36 )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 128LL))(
             *(_QWORD *)(a1 + 8),
             a1,
             a2,
             v5,
             v9,
             a3,
             v8);
  v6 = 0;
  if ( *(_DWORD *)(a2 + 12) != 1 )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 128LL))(
             *(_QWORD *)(a1 + 8),
             a1,
             a2,
             v5,
             v9,
             a3,
             v8);
  return v6;
}
