// Function: sub_D4B410
// Address: 0xd4b410
//
__int64 __fastcall sub_D4B410(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v19; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 v21; // [rsp+18h] [rbp-88h]
  _QWORD v22[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23; // [rsp+30h] [rbp-70h]
  int v24; // [rsp+38h] [rbp-68h]
  __int64 v25; // [rsp+40h] [rbp-60h]
  __int64 v26; // [rsp+48h] [rbp-58h]
  _BYTE *v27; // [rsp+50h] [rbp-50h]
  __int64 v28; // [rsp+58h] [rbp-48h]
  _BYTE v29[64]; // [rsp+60h] [rbp-40h] BYREF

  if ( !(unsigned __int8)sub_D4B3D0(a1) )
    return 0;
  v3 = **(_QWORD **)(a1 + 32);
  v4 = sub_D48970(a1);
  if ( !v4 )
    return 0;
  v20 = *(_QWORD *)(v4 - 64);
  v19 = *(_QWORD *)(v4 - 32);
  v5 = sub_AA5930(v3);
  v21 = v6;
  if ( v6 == v5 )
    return 0;
  v7 = v5;
  while ( 1 )
  {
    v9 = a1;
    v22[0] = 6;
    v22[1] = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = v29;
    v28 = 0x200000000LL;
    if ( !(unsigned __int8)sub_10238A0(v7, a1, a2, v22, 0, 0) )
    {
      if ( v27 != v29 )
        _libc_free(v27, a1);
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD60C0(v22);
      if ( !v7 )
        BUG();
      goto LABEL_20;
    }
    v10 = sub_D47930(a1);
    v11 = *(_QWORD *)(v7 - 8);
    v12 = v10;
    if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
    {
      v13 = 0;
      v9 = v11 + 32LL * *(unsigned int *)(v7 + 72);
      while ( v12 != *(_QWORD *)(v9 + 8 * v13) )
      {
        if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == (_DWORD)++v13 )
          goto LABEL_26;
      }
      v14 = 32 * v13;
    }
    else
    {
LABEL_26:
      v14 = 0x1FFFFFFFE0LL;
    }
    v15 = *(_QWORD *)(v11 + v14);
    if ( v20 == v15 )
      break;
    v9 = v19;
    if ( v19 == v15 || v7 == v19 || v7 == v20 )
      break;
    if ( v27 != v29 )
      _libc_free(v27, v19);
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
      sub_BD60C0(v22);
LABEL_20:
    v16 = *(_QWORD *)(v7 + 32);
    if ( !v16 )
      BUG();
    v7 = 0;
    if ( *(_BYTE *)(v16 - 24) == 84 )
      v7 = v16 - 24;
    if ( v21 == v7 )
      return 0;
  }
  v17 = v7;
  if ( v27 != v29 )
    _libc_free(v27, v9);
  if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
    sub_BD60C0(v22);
  return v17;
}
