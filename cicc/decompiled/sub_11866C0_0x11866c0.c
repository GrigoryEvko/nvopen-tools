// Function: sub_11866C0
// Address: 0x11866c0
//
_QWORD *__fastcall sub_11866C0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  _QWORD *result; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  unsigned int v9; // edx
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned __int8 **v15; // r12
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rdx
  _BYTE *v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rax
  __int64 v25; // [rsp-80h] [rbp-80h]
  __int64 v26[2]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v27; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v28[3]; // [rsp-60h] [rbp-60h] BYREF
  __int16 v29; // [rsp-48h] [rbp-48h]

  if ( (*(_WORD *)(a1 + 2) & 0x3Fu) - 32 > 1 )
    return 0;
  v4 = a2;
  v5 = (unsigned int)sub_BCB060(*(_QWORD *)(a2 + 8));
  if ( !(unsigned __int8)sub_1178DE0(*(_QWORD *)(a1 - 32)) )
    return 0;
  if ( (*(_WORD *)(a1 + 2) & 0x3F) == 0x21 )
  {
    v4 = (__int64)a3;
    a3 = (_BYTE *)a2;
    if ( *(_BYTE *)a2 != 59 )
      return 0;
  }
  else if ( *a3 != 59 )
  {
    return 0;
  }
  v7 = *((_QWORD *)a3 - 8);
  if ( !v7 )
    return 0;
  v8 = *((_QWORD *)a3 - 4);
  if ( !v8 )
    BUG();
  if ( *(_BYTE *)v8 != 17 )
  {
    v21 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v21 > 1 )
      return 0;
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    v22 = sub_AD7630(v8, 0, v21);
    v8 = (__int64)v22;
    if ( !v22 || *v22 != 17 )
      return 0;
  }
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 > 0x40 )
  {
    if ( v9 - (unsigned int)sub_C444A0(v8 + 24) > 0x40 )
      return 0;
    v10 = **(_QWORD **)(v8 + 24);
  }
  else
  {
    v10 = *(_QWORD *)(v8 + 24);
  }
  if ( (_DWORD)v5 - 1 != v10 )
    return 0;
  if ( *(_BYTE *)v7 != 85 )
    return 0;
  v11 = *(_QWORD *)(v7 - 32);
  if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(v7 + 80) || *(_DWORD *)(v11 + 36) != 65 )
    return 0;
  if ( v4 != v7 )
  {
    if ( *(_BYTE *)v4 != 17 )
    {
      v23 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
      if ( (unsigned int)v23 > 1 )
        return 0;
      if ( *(_BYTE *)v4 > 0x15u )
        return 0;
      v24 = sub_AD7630(v4, 0, v23);
      v4 = (__int64)v24;
      if ( !v24 || *v24 != 17 )
        return 0;
    }
    v12 = *(_DWORD *)(v4 + 32);
    if ( v12 > 0x40 )
    {
      if ( v12 - (unsigned int)sub_C444A0(v4 + 24) > 0x40 )
        return 0;
      v13 = **(_QWORD **)(v4 + 24);
    }
    else
    {
      v13 = *(_QWORD *)(v4 + 24);
    }
    if ( v5 != v13 )
      return 0;
  }
  v14 = *(_QWORD *)(a1 - 64);
  v28[0] = 0;
  v27 = v14;
  v28[1] = v14;
  v15 = *(unsigned __int8 ***)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v15 != 57 )
    return 0;
  v16 = v14;
  if ( (unsigned __int8 *)v14 == *(v15 - 8) )
  {
    if ( sub_99C280((__int64)v28, 15, *(v15 - 4)) )
      goto LABEL_27;
    v16 = v27;
  }
  if ( *(v15 - 4) != (unsigned __int8 *)v16 || !sub_99C280((__int64)v28, 15, *(v15 - 8)) )
    return 0;
LABEL_27:
  v17 = 0;
  v27 = *(_QWORD *)(v7 + 8);
  v18 = (__int64 *)sub_B43CA0(v7);
  v19 = sub_B6E160(v18, 0x43u, (__int64)&v27, 1);
  v26[0] = v14;
  v29 = 257;
  v20 = v19;
  v26[1] = *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
  if ( v19 )
    v17 = *(_QWORD *)(v19 + 24);
  result = sub_BD2CC0(88, 3u);
  if ( result )
  {
    v25 = (__int64)result;
    sub_B44260((__int64)result, **(_QWORD **)(v17 + 16), 56, 3u, 0, 0);
    *(_QWORD *)(v25 + 72) = 0;
    sub_B4A290(v25, v17, v20, v26, 2, (__int64)&v27, 0, 0);
    return (_QWORD *)v25;
  }
  return result;
}
