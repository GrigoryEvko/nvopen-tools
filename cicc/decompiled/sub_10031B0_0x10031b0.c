// Function: sub_10031B0
// Address: 0x10031b0
//
__int64 __fastcall sub_10031B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  unsigned __int8 v6; // al
  __int64 v7; // r15
  bool v8; // r14
  __int64 v10; // r8
  __int64 v11; // rdx
  _QWORD *v12; // r12
  unsigned int v13; // r14d
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  int v16; // eax
  __int64 v17; // [rsp+0h] [rbp-50h]
  unsigned int v18; // [rsp+Ch] [rbp-44h]
  _BYTE *v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v4 = 0;
  v6 = *(_BYTE *)a3;
  if ( *(_BYTE *)a1 <= 0x15u )
    v4 = a1;
  v7 = v4;
  if ( *(_BYTE *)a2 > 0x15u )
  {
    if ( v6 > 0x15u )
    {
      v19 = 0;
      v8 = 0;
      goto LABEL_11;
    }
    v19 = 0;
  }
  else
  {
    v8 = v4 != 0;
    if ( v6 > 0x15u )
    {
      v19 = (_BYTE *)a2;
      goto LABEL_11;
    }
    if ( v4 )
      return sub_AD5A90(v4, (_BYTE *)a2, (unsigned __int8 *)a3, 0);
    v19 = (_BYTE *)a2;
  }
  if ( v6 == 17 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v8 = 0;
    if ( *(_BYTE *)(v10 + 8) != 17 )
      goto LABEL_11;
    v13 = *(_DWORD *)(a3 + 32);
    v14 = *(unsigned int *)(v10 + 32);
    if ( v13 > 0x40 )
    {
      v17 = a4;
      v18 = *(_DWORD *)(v10 + 32);
      v21 = *(_QWORD *)(a1 + 8);
      v16 = sub_C444A0(a3 + 24);
      v10 = v21;
      v14 = v18;
      a4 = v17;
      if ( v13 - v16 > 0x40 )
        return sub_ACADE0((__int64 **)v10);
      v15 = **(_QWORD **)(a3 + 24);
    }
    else
    {
      v15 = *(_QWORD *)(a3 + 24);
    }
    if ( v14 <= v15 )
      return sub_ACADE0((__int64 **)v10);
  }
  v8 = 0;
LABEL_11:
  v20 = a4;
  if ( (unsigned __int8)sub_1003090(a4, (unsigned __int8 *)a3) )
    return sub_ACADE0(*(__int64 ***)(a1 + 8));
  if ( *(_BYTE *)a2 == 13
    || (unsigned __int8)sub_1003090(v20, (unsigned __int8 *)a2) && sub_98ED70((unsigned __int8 *)a1, 0, 0, 0, 0) )
  {
    return a1;
  }
  if ( v8 && v19 == sub_AD7630(v7, 0, v11) )
    return a1;
  if ( *(_BYTE *)a2 == 90 )
  {
    v12 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
        ? *(_QWORD **)(a2 - 8)
        : (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( a1 == *v12 && a3 == v12[4] )
      return a1;
  }
  return 0;
}
