// Function: sub_283ED10
// Address: 0x283ed10
//
__int64 __fastcall sub_283ED10(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 *v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  bool v14; // zf
  __int64 v16; // [rsp-8h] [rbp-D8h]
  __int64 *v17; // [rsp+8h] [rbp-C8h]
  __int64 v19; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v20[8]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-98h]
  char v22; // [rsp+4Ch] [rbp-84h]
  unsigned __int64 v23; // [rsp+68h] [rbp-68h]
  char v24; // [rsp+7Ch] [rbp-54h]
  char v25; // [rsp+90h] [rbp-40h]

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  v9 = sub_22D3D20(a4, &qword_4F8A320, a3, a5);
  v10 = *(__int64 **)(a2 + 72);
  v19 = *(_QWORD *)(v9 + 8);
  v17 = *(__int64 **)(a2 + 80);
  if ( v10 == v17 )
    return a1;
  while ( 1 )
  {
    while ( 1 )
    {
      sub_283EB30(v20, a3, v10, a4, a5, a6, &v19);
      if ( v25 )
        break;
LABEL_3:
      if ( v17 == ++v10 )
        return a1;
    }
    if ( *(_BYTE *)(a6 + 24) )
      break;
    sub_22D08B0(a4, (__int64)a3, (__int64)v20);
    sub_BBADB0(a1, (__int64)v20, v12, v13);
    v14 = v25 == 0;
    *(_QWORD *)(a6 + 32) = *a3;
    if ( v14 )
      goto LABEL_3;
    if ( !v24 )
      _libc_free(v23);
    if ( v22 )
      goto LABEL_3;
    ++v10;
    _libc_free(v21);
    if ( v17 == v10 )
      return a1;
  }
  sub_BBADB0(a1, (__int64)v20, v16, v11);
  if ( v25 )
  {
    if ( !v24 )
      _libc_free(v23);
    if ( !v22 )
      _libc_free(v21);
  }
  return a1;
}
