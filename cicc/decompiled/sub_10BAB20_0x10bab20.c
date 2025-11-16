// Function: sub_10BAB20
// Address: 0x10bab20
//
__int64 __fastcall sub_10BAB20(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // edx
  unsigned int v6; // r12d
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  const void **v16; // r15
  __int64 v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]

  v24 = 36;
  if ( !a1 )
    return 0;
  v4 = sub_B53900(a1);
  v22 = sub_B53630(v4, v24);
  v6 = v5;
  v23 = v5;
  if ( !(_BYTE)v5 )
    return 0;
  v8 = *(_BYTE **)(a1 - 64);
  if ( *v8 != 42 )
    return 0;
  v9 = *((_QWORD *)v8 - 8);
  if ( !v9 )
    return 0;
  *a2 = v9;
  v10 = *((_QWORD *)v8 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = v10 + 24;
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v10 + 24) == 1 )
        goto LABEL_17;
    }
    else
    {
      v12 = *(_QWORD *)(v10 + 24);
      if ( v12 )
      {
        v9 = v12 - 1;
        if ( (v12 & (v12 - 1)) == 0 )
          goto LABEL_17;
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17 > 1 )
      return 0;
  }
  else
  {
    v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v10 > 0x15u )
      return 0;
  }
  v13 = sub_AD7630(v10, 1, v9);
  if ( !v13 || *v13 != 17 )
    return 0;
  v11 = (__int64)(v13 + 24);
  if ( *((_DWORD *)v13 + 8) > 0x40u )
  {
    if ( (unsigned int)sub_C44630((__int64)(v13 + 24)) != 1 )
      return 0;
  }
  else
  {
    v14 = *((_QWORD *)v13 + 3);
    if ( !v14 )
      return 0;
    v9 = v14 - 1;
    if ( (v14 & (v14 - 1)) != 0 )
      return 0;
  }
LABEL_17:
  v15 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v15 == 17 )
  {
    v16 = (const void **)(v15 + 24);
    if ( *(_DWORD *)(v15 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v15 + 24) == 1 )
        goto LABEL_27;
    }
    else
    {
      v17 = *(_QWORD *)(v15 + 24);
      if ( v17 )
      {
        v9 = v17 - 1;
        if ( (v17 & (v17 - 1)) == 0 )
          goto LABEL_27;
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17 > 1 )
      return 0;
  }
  else
  {
    v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v15 > 0x15u )
      return 0;
  }
  v18 = sub_AD7630(v15, 1, v9);
  if ( !v18 || *v18 != 17 )
    return 0;
  v16 = (const void **)(v18 + 24);
  if ( *((_DWORD *)v18 + 8) > 0x40u )
  {
    if ( (unsigned int)sub_C44630((__int64)(v18 + 24)) != 1 )
      return 0;
  }
  else
  {
    v19 = *((_QWORD *)v18 + 3);
    if ( !v19 || (v19 & (v19 - 1)) != 0 )
      return 0;
  }
LABEL_27:
  if ( (int)sub_C49970((__int64)v16, (unsigned __int64 *)v11) <= 0 )
    return 0;
  v20 = *(_DWORD *)(v11 + 8);
  v23 = v20;
  if ( v20 > 0x40 )
  {
    sub_C43780((__int64)&v22, (const void **)v11);
    v20 = v23;
    if ( v23 <= 0x40 )
      goto LABEL_30;
    sub_C47690((__int64 *)&v22, 1u);
    if ( v23 <= 0x40 )
      goto LABEL_33;
    if ( sub_C43C50((__int64)&v22, v16) )
    {
      if ( v22 )
        j_j___libc_free_0_0(v22);
      goto LABEL_34;
    }
    if ( v22 )
      j_j___libc_free_0_0(v22);
    return 0;
  }
  v22 = *(_QWORD *)v11;
LABEL_30:
  v21 = 0;
  if ( v20 >= 2 )
    v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & (2 * v22);
  v22 = v21;
LABEL_33:
  if ( (const void *)v22 != *v16 )
    return 0;
LABEL_34:
  if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v11 + 8) <= 0x40u )
  {
    *(_QWORD *)a3 = *(_QWORD *)v11;
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v11 + 8);
  }
  else
  {
    sub_C43990(a3, v11);
  }
  return v6;
}
