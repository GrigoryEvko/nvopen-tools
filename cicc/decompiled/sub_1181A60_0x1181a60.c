// Function: sub_1181A60
// Address: 0x1181a60
//
__int64 __fastcall sub_1181A60(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // r14
  __int64 v7; // rsi
  _BYTE *v8; // rdi
  _QWORD v9[2]; // [rsp+0h] [rbp-E0h] BYREF
  _BYTE *v10; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-C8h]
  _BYTE v12[64]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v13; // [rsp+60h] [rbp-80h] BYREF
  char *v14; // [rsp+68h] [rbp-78h]
  __int64 v15; // [rsp+70h] [rbp-70h]
  int v16; // [rsp+78h] [rbp-68h]
  char v17; // [rsp+7Ch] [rbp-64h]
  char v18; // [rsp+80h] [rbp-60h] BYREF

  v2 = 0;
  if ( *(_BYTE *)a2 != 85 )
    return v2;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 )
    return v2;
  if ( *(_BYTE *)v4 )
    return v2;
  if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) )
    return v2;
  if ( *(_DWORD *)(v4 + 36) != *(_DWORD *)a1 )
    return v2;
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - v5)) != *(_QWORD *)(a1 + 32) )
    return v2;
  v2 = 1;
  v6 = *(_BYTE **)(a2 + 32 * (*(unsigned int *)(a1 + 40) - v5));
  if ( (unsigned __int8)(*v6 - 12) <= 1u )
    return v2;
  if ( (unsigned __int8)(*v6 - 9) > 2u )
    return sub_1178DE0((__int64)v6);
  v7 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 40) - v5));
  v17 = 1;
  v14 = &v18;
  v11 = 0x800000000LL;
  v13 = 0;
  v15 = 8;
  v16 = 0;
  v10 = v12;
  v9[0] = &v13;
  v9[1] = &v10;
  v2 = sub_AA8FD0(v9, (__int64)v6);
  if ( (_BYTE)v2 )
  {
    while ( 1 )
    {
      v8 = v10;
      if ( !(_DWORD)v11 )
        break;
      v7 = *(_QWORD *)&v10[8 * (unsigned int)v11 - 8];
      LODWORD(v11) = v11 - 1;
      if ( !(unsigned __int8)sub_AA8FD0(v9, v7) )
        goto LABEL_20;
    }
  }
  else
  {
LABEL_20:
    v8 = v10;
    v2 = 0;
  }
  if ( v8 != v12 )
    _libc_free(v8, v7);
  if ( !v17 )
    _libc_free(v14, v7);
  if ( !(_BYTE)v2 )
    return sub_1178DE0((__int64)v6);
  else
    return v2;
}
