// Function: sub_13D69B0
// Address: 0x13d69b0
//
__int64 __fastcall sub_13D69B0(unsigned __int8 *a1, unsigned __int8 *a2, char a3, _QWORD *a4)
{
  __int64 v6; // r12
  unsigned __int8 *v7; // rbx
  __int64 v8; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  _BYTE v14[8]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 *v15; // [rsp+8h] [rbp-48h]
  _BYTE v16[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]

  v6 = (__int64)a2;
  v7 = a1;
  if ( a1[16] <= 0x10u )
  {
    if ( a2[16] > 0x10u )
    {
      v7 = a2;
      v6 = (__int64)a1;
    }
    else
    {
      v8 = sub_14D6F90(12, a1, a2, *a4);
      if ( v8 )
        return v8;
      if ( a1[16] == 9 )
        goto LABEL_7;
    }
  }
  if ( *(_BYTE *)(v6 + 16) == 9 )
  {
LABEL_7:
    v8 = sub_15A11D0(*(_QWORD *)v7, 0, 0);
    goto LABEL_8;
  }
  v8 = sub_13CDA40(v7, (_QWORD *)v6);
LABEL_8:
  if ( v8 )
    return v8;
  if ( (unsigned __int8)sub_13CC390(v6)
    || (unsigned __int8)sub_13CBF20(v6) && ((a3 & 8) != 0 || (unsigned __int8)sub_14AB3F0(v7, a4[1], 0)) )
  {
    return (__int64)v7;
  }
  if ( (a3 & 2) == 0 )
    return v8;
  v17 = v6;
  if ( !sub_13D66D0((__int64)v16, (__int64)v7, v10, v11) )
  {
    v15 = v7;
    if ( !sub_13D66D0((__int64)v14, v6, v12, v13) )
      return v8;
  }
  return sub_15A06D0(*(_QWORD *)v7);
}
