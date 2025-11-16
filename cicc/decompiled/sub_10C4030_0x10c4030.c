// Function: sub_10C4030
// Address: 0x10c4030
//
__int64 __fastcall sub_10C4030(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdx
  unsigned __int8 *v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  _BYTE *v10; // rax
  _BYTE *v11; // r13
  __int64 v12; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)a2 - 54) > 1u )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v5 = *(unsigned __int8 **)v4;
  v6 = **(_BYTE **)v4;
  if ( v6 != 17 )
  {
    v4 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v5 + 1) + 8LL) - 17;
    if ( (unsigned int)v4 > 1 || v6 > 0x15u )
      return 0;
LABEL_23:
    v10 = sub_AD7630((__int64)v5, 1, v4);
    if ( !v10 || *v10 != 17 )
      return 0;
    v11 = v10 + 24;
    if ( *((_DWORD *)v10 + 8) > 0x40u )
    {
      if ( (unsigned int)sub_C44630((__int64)(v10 + 24)) != 1 )
        return 0;
    }
    else
    {
      v12 = *((_QWORD *)v10 + 3);
      if ( !v12 || (v12 & (v12 - 1)) != 0 )
        return 0;
    }
    **a1 = v11;
    goto LABEL_12;
  }
  if ( *((_DWORD *)v5 + 8) <= 0x40u )
  {
    v7 = *((_QWORD *)v5 + 3);
    if ( v7 )
    {
      v4 = v7 - 1;
      if ( (v7 & (v7 - 1)) == 0 )
        goto LABEL_11;
    }
    goto LABEL_17;
  }
  if ( (unsigned int)sub_C44630((__int64)(v5 + 24)) != 1 )
  {
LABEL_17:
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v5 + 1) + 8LL) - 17 > 1 )
      return 0;
    goto LABEL_23;
  }
LABEL_11:
  **a1 = v5 + 24;
LABEL_12:
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v8 = *(_QWORD *)(a2 - 8);
  else
    v8 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v9 = *(_QWORD *)(v8 + 32);
  if ( v9 )
  {
    *a1[1] = v9;
    return 1;
  }
  return 0;
}
