// Function: sub_1007280
// Address: 0x1007280
//
__int64 __fastcall sub_1007280(_QWORD **a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax

  if ( *(_BYTE *)a2 == 17 )
  {
    if ( *(_DWORD *)(a2 + 32) <= 0x40u )
    {
      v7 = *(_QWORD *)(a2 + 24);
      if ( v7 )
      {
        a3 = v7 - 1;
        if ( (v7 & (v7 - 1)) == 0 )
          goto LABEL_4;
      }
    }
    else if ( (unsigned int)sub_C44630(a2 + 24) == 1 )
    {
LABEL_4:
      **a1 = a2 + 24;
      return 1;
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
      return 0;
  }
  else
  {
    a3 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)a3 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
  }
  v4 = sub_AD7630(a2, 1, a3);
  if ( !v4 || *v4 != 17 )
    return 0;
  v5 = v4 + 24;
  if ( *((_DWORD *)v4 + 8) > 0x40u )
  {
    if ( (unsigned int)sub_C44630((__int64)(v4 + 24)) == 1 )
      goto LABEL_17;
    return 0;
  }
  v6 = *((_QWORD *)v4 + 3);
  if ( !v6 || (v6 & (v6 - 1)) != 0 )
    return 0;
LABEL_17:
  **a1 = v5;
  return 1;
}
