// Function: sub_11E0900
// Address: 0x11e0900
//
__int64 __fastcall sub_11E0900(__int64 a1, __int64 a2)
{
  int v3; // eax
  char v4; // r13
  char v5; // dl
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // [rsp+0h] [rbp-40h] BYREF
  __int64 v9; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v10; // [rsp+10h] [rbp-30h] BYREF
  __int64 v11; // [rsp+18h] [rbp-28h]

  v3 = *(_DWORD *)(a2 + 4);
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v4 = sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v3 & 0x7FFFFFF)), &v8, 1u);
  v5 = sub_98B0F0(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), &v10, 1u);
  if ( v4 && !v9 )
    return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)&v10);
  result = 0;
  if ( v5 )
  {
    if ( !v11 )
      return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)&v10);
    if ( v4 )
    {
      v7 = sub_C935B0(&v8, v10, v11, 0);
      if ( v7 == -1 )
        v7 = v9;
      return sub_AD64C0(*(_QWORD *)(a2 + 8), v7, 0);
    }
  }
  return result;
}
