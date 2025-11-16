// Function: sub_11E9C60
// Address: 0x11e9c60
//
unsigned __int64 __fastcall sub_11E9C60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // r15
  const char *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  int v12; // r14d
  const char *v13; // rax
  unsigned __int64 v14; // rdx
  unsigned int v15; // esi
  unsigned __int64 result; // rax
  __int64 v17; // [rsp+8h] [rbp-68h]
  _BYTE v18[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  v5 = sub_B43CA0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  v7 = (__int64 *)v5;
  if ( v6 )
  {
    if ( *(_BYTE *)v6 )
    {
      v6 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v6 + 24) )
    {
      v6 = 0;
    }
  }
  v8 = sub_BD5D20(v6);
  if ( v11 != 4
    || *(_DWORD *)v8 != 1852403046 && *(_DWORD *)v8 != 2019650918
    || !(unsigned __int8)sub_11E9B60(a1, v7, (__int64)v8, 4u, v9, v10)
    || (result = sub_11DB650(a2, a3, 1, *(__int64 **)(a1 + 24), 0)) == 0 )
  {
    v12 = sub_B45210(a2) | 8;
    v13 = sub_BD5D20(v6);
    if ( v14 <= 3 || (v15 = 248, *(_DWORD *)v13 != 1852403046) )
      v15 = 237;
    LODWORD(v17) = v12;
    v19 = 257;
    BYTE4(v17) = 1;
    result = sub_B33C40(
               a3,
               v15,
               *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
               *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
               v17,
               (__int64)v18);
    if ( result )
    {
      if ( *(_BYTE *)result == 85 )
        *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
    }
  }
  return result;
}
