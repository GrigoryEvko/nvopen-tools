// Function: sub_AB2160
// Address: 0xab2160
//
__int64 __fastcall sub_AB2160(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // r14
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF760(a3) )
  {
    v6 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v6;
    if ( v6 > 0x40 )
      sub_C43780(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v7 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 > 0x40 )
      sub_C43780(a1 + 16, a2 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    return a1;
  }
  if ( !sub_AAF7D0(a3) && !sub_AAF760(a2) )
  {
    if ( !sub_AB0100(a2) && sub_AB0100(a3) )
    {
      sub_AB2160(a1, a3, a2, a4);
      return a1;
    }
    if ( !sub_AB0100(a2) && !sub_AB0100(a3) )
    {
      if ( (int)sub_C49970(a2, a3) >= 0 )
      {
        v11 = a3 + 16;
        if ( (int)sub_C49970(a2 + 16, v11) >= 0 )
        {
          if ( (int)sub_C49970(a2, v11) >= 0 )
          {
LABEL_25:
            sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
            return a1;
          }
LABEL_38:
          sub_9865C0((__int64)v14, v11);
          sub_9865C0((__int64)v13, a2);
          sub_AADC30(a1, (__int64)v13, v14);
          sub_969240(v13);
          sub_969240(v14);
          return a1;
        }
LABEL_43:
        sub_AAF450(a1, a2);
        return a1;
      }
      if ( (int)sub_C49970(a2 + 16, a3) <= 0 )
        goto LABEL_25;
      if ( (int)sub_C49970(a2 + 16, a3 + 16) < 0 )
      {
LABEL_30:
        sub_9865C0((__int64)v14, a2 + 16);
        sub_9865C0((__int64)v13, a3);
        sub_AADC30(a1, (__int64)v13, v14);
        sub_969240(v13);
        sub_969240(v14);
        return a1;
      }
LABEL_33:
      sub_AAF450(a1, a3);
      return a1;
    }
    if ( !sub_AB0100(a2) || sub_AB0100(a3) )
    {
      v12 = a3 + 16;
      if ( (int)sub_C49970(a3 + 16, a2 + 16) < 0 )
      {
        if ( (int)sub_C49970(a3, a2 + 16) >= 0 )
        {
          if ( (int)sub_C49970(a3, a2) < 0 )
          {
            sub_9865C0((__int64)v14, v12);
            sub_9865C0((__int64)v13, a2);
            sub_AADC30(a1, (__int64)v13, v14);
            sub_969240(v13);
            sub_969240(v14);
            return a1;
          }
          goto LABEL_33;
        }
      }
      else if ( (int)sub_C49970(v12, a2) <= 0 )
      {
        if ( (int)sub_C49970(a3, a2) < 0 )
          goto LABEL_43;
        goto LABEL_30;
      }
    }
    else
    {
      if ( (int)sub_C49970(a3, a2 + 16) >= 0 )
      {
        if ( (int)sub_C49970(a3, a2) < 0 )
        {
          v11 = a3 + 16;
          if ( (int)sub_C49970(v11, a2) <= 0 )
            goto LABEL_25;
          goto LABEL_38;
        }
        goto LABEL_33;
      }
      if ( (int)sub_C49970(a3 + 16, a2 + 16) < 0 )
        goto LABEL_33;
      if ( (int)sub_C49970(a3 + 16, a2) <= 0 )
        goto LABEL_30;
    }
    sub_AB0360(a1, a2, a3, a4);
    return a1;
  }
  v9 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 8) = v9;
  if ( v9 > 0x40 )
    sub_C43780(a1, a3);
  else
    *(_QWORD *)a1 = *(_QWORD *)a3;
  v10 = *(_DWORD *)(a3 + 24);
  *(_DWORD *)(a1 + 24) = v10;
  if ( v10 > 0x40 )
    sub_C43780(a1 + 16, a3 + 16);
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
  return a1;
}
