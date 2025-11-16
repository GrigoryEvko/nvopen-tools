// Function: sub_12FC8E0
// Address: 0x12fc8e0
//
__int64 __fastcall sub_12FC8E0(__int64 *a1, _QWORD *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v7; // r13
  int *v8; // rax
  int *v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *a1;
  v8 = __errno_location();
  *v8 = 0;
  v9 = v8;
  v10 = sub_130AAB0(v7, v19, 0);
  if ( *v9 )
    return 1;
  v11 = v19[0];
  if ( *(_BYTE *)v19[0] != 45 )
    return 1;
  *a3 = v10;
  v13 = sub_130AAB0(v11 + 1, v19, 0);
  if ( *v9 )
    return 1;
  if ( *(_BYTE *)v19[0] != 58 )
    return 1;
  v14 = v19[0] + 1LL;
  *a4 = v13;
  v15 = sub_130AAB0(v14, v19, 0);
  if ( *v9 )
    return 1;
  *a5 = v15;
  v16 = (*(_BYTE *)v19[0] == 124) + v19[0];
  *a2 -= v16 - *a1;
  *a1 = v16;
  return 0;
}
