// Function: sub_11AF6F0
// Address: 0x11af6f0
//
_QWORD *__fastcall sub_11AF6F0(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *result; // rax
  __int64 v3; // r13
  unsigned int v4; // r14d
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r15
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // r15
  int v14; // r9d
  unsigned __int64 v15; // r8
  _DWORD *v16; // rax
  _BYTE *v17; // rdx
  _DWORD *i; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // edx
  _BYTE *v22; // rdi
  _BYTE *v23; // r14
  __int64 v24; // r12
  __int64 v25; // r8
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // [rsp+8h] [rbp-B8h]
  _QWORD *v29; // [rsp+8h] [rbp-B8h]
  _QWORD *v30; // [rsp+8h] [rbp-B8h]
  _BYTE v31[32]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v32; // [rsp+30h] [rbp-90h]
  _BYTE *v33; // [rsp+40h] [rbp-80h] BYREF
  __int64 v34; // [rsp+48h] [rbp-78h]
  _BYTE v35[112]; // [rsp+50h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 - 96);
  if ( *(_BYTE *)v1 != 92
    || **(_BYTE **)(v1 - 32) != 13
    || !(unsigned __int8)sub_B4F5D0(*(_QWORD *)(a1 - 96)) && !(unsigned __int8)sub_B4F540(v1) )
  {
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v1 + 8) + 8LL) == 18 )
    return 0;
  v3 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v3 != 17 )
    return 0;
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 > 0x40 )
  {
    if ( v4 - (unsigned int)sub_C444A0(v3 + 24) > 0x40 )
      return 0;
    v5 = **(_QWORD **)(v3 + 24);
  }
  else
  {
    v5 = *(_QWORD *)(v3 + 24);
  }
  v6 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v6 != 90 )
    return 0;
  v7 = (*(_BYTE *)(v6 + 7) & 0x40) != 0
     ? *(_QWORD **)(v6 - 8)
     : (_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  v8 = *(_QWORD *)(v1 - 64);
  if ( v8 != *v7 )
    return 0;
  v9 = v7[4];
  if ( *(_BYTE *)v9 != 17 )
  {
    v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
    if ( (unsigned int)v26 > 1 )
      return 0;
    if ( *(_BYTE *)v9 > 0x15u )
      return 0;
    v27 = sub_AD7630(v9, 0, v26);
    v9 = (__int64)v27;
    if ( !v27 || *v27 != 17 )
      return 0;
  }
  v10 = *(_DWORD *)(v9 + 32);
  if ( v10 > 0x40 )
  {
    if ( v10 - (unsigned int)sub_C444A0(v9 + 24) > 0x40 )
      return 0;
    v11 = **(_QWORD **)(v9 + 24);
  }
  else
  {
    v11 = *(_QWORD *)(v9 + 24);
  }
  if ( v5 != v11 )
    return 0;
  v12 = *(_QWORD *)(v1 + 8);
  v33 = v35;
  v13 = *(unsigned int *)(v12 + 32);
  v34 = 0x1000000000LL;
  v14 = v13;
  v15 = v13;
  if ( v13 )
  {
    v16 = v35;
    v17 = v35;
    if ( v13 > 0x10 )
    {
      sub_C8D5F0((__int64)&v33, v35, v13, 4u, v13, v13);
      v17 = v33;
      v14 = v13;
      v16 = &v33[4 * (unsigned int)v34];
    }
    for ( i = &v17[4 * v13]; i != v16; ++v16 )
    {
      if ( v16 )
        *v16 = 0;
    }
    LODWORD(v34) = v14;
    v19 = *(_QWORD *)(v1 + 72);
    v20 = 0;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v19 + 4 * v20);
      v22 = v33;
      if ( v20 == v5 )
      {
        if ( (_DWORD)v20 == v21 )
        {
          result = 0;
          goto LABEL_39;
        }
        *(_DWORD *)&v33[4 * v20] = v20;
      }
      else
      {
        *(_DWORD *)&v33[4 * v20] = v21;
      }
      if ( ++v20 == v13 )
      {
        v23 = v33;
        v15 = (unsigned int)v34;
        goto LABEL_37;
      }
    }
  }
  v23 = v35;
LABEL_37:
  v28 = v15;
  v24 = *(_QWORD *)(v1 - 32);
  v32 = 257;
  v19 = unk_3F1FE60;
  result = sub_BD2C40(112, unk_3F1FE60);
  v25 = v28;
  if ( result )
  {
    v19 = v8;
    v29 = result;
    sub_B4E9E0((__int64)result, v8, v24, v23, v25, (__int64)v31, 0, 0);
    v22 = v33;
    result = v29;
  }
  else
  {
    v22 = v33;
  }
LABEL_39:
  if ( v22 != v35 )
  {
    v30 = result;
    _libc_free(v22, v19);
    return v30;
  }
  return result;
}
