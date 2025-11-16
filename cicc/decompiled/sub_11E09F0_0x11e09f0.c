// Function: sub_11E09F0
// Address: 0x11e09f0
//
__int64 __fastcall sub_11E09F0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  char v6; // r15
  char v7; // al
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v12; // [rsp+10h] [rbp-40h] BYREF
  __int64 v13; // [rsp+18h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 4);
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v6 = sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v5 & 0x7FFFFFF)), &v10, 1u);
  v7 = sub_98B0F0(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), &v12, 1u);
  if ( v6 )
  {
    if ( !v11 )
      return sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)&v12);
    if ( v7 )
    {
      v8 = sub_C934D0(&v10, v12, v13, 0);
      if ( v8 == -1 )
        v8 = v11;
      return sub_AD64C0(*(_QWORD *)(a2 + 8), v8, 0);
    }
    return 0;
  }
  if ( !v7 )
    return 0;
  if ( v13 )
    return 0;
  result = sub_11CA050(
             *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
             a3,
             *(_QWORD *)(a1 + 16),
             *(__int64 **)(a1 + 24));
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
