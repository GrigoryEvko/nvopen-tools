// Function: sub_23FAB40
// Address: 0x23fab40
//
__int64 __fastcall sub_23FAB40(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v4; // r12d
  unsigned __int64 v5; // r14
  int v7; // eax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_BC8C50(a1, &v9, v10);
  if ( !(_BYTE)v4 )
    return v4;
  v5 = v10[0] + v9;
  if ( !(v10[0] + v9) )
    return 0;
  v7 = sub_F02DD0(v9, v10[0] + v9);
  v8 = v10[0];
  *a2 = v7;
  *a3 = sub_F02DD0(v8, v5);
  return v4;
}
