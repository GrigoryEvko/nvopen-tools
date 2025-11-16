// Function: sub_1110D90
// Address: 0x1110d90
//
__int64 __fastcall sub_1110D90(__int64 **a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4; // r12
  unsigned int v5; // r14d
  __int64 v6; // rax
  _BYTE *v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // r14
  __int64 v10; // rax
  int v11; // r13d
  int v12; // r12d

  v2 = *(_QWORD *)(a2 - 64);
  if ( (unsigned __int8)(*(_BYTE *)v2 - 42) > 0x11u )
    return 0;
  **a1 = v2;
  v4 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v4 == 17 )
  {
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 > 0x40 )
    {
      v11 = sub_C445E0(v4 + 24);
      if ( v11 && v5 == (unsigned int)sub_C444A0(v4 + 24) + v11 )
        goto LABEL_21;
    }
    else
    {
      v6 = *(_QWORD *)(v4 + 24);
      if ( v6 )
      {
        v2 = v6 + 1;
        if ( (v6 & (v6 + 1)) == 0 )
        {
LABEL_21:
          *a1[1] = v4 + 24;
          return 1;
        }
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17 > 1 )
      return 0;
  }
  else
  {
    v2 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17;
    if ( (unsigned int)v2 > 1 || *(_BYTE *)v4 > 0x15u )
      return 0;
  }
  v7 = sub_AD7630(v4, 1, v2);
  if ( v7 && *v7 == 17 )
  {
    v8 = *((_DWORD *)v7 + 8);
    v9 = (__int64)(v7 + 24);
    if ( v8 > 0x40 )
    {
      v12 = sub_C445E0((__int64)(v7 + 24));
      if ( v12 && v8 == (unsigned int)sub_C444A0(v9) + v12 )
        goto LABEL_12;
    }
    else
    {
      v10 = *((_QWORD *)v7 + 3);
      if ( v10 && (v10 & (v10 + 1)) == 0 )
      {
LABEL_12:
        *a1[1] = v9;
        return 1;
      }
    }
  }
  return 0;
}
