// Function: sub_8D1330
// Address: 0x8d1330
//
__int64 __fastcall sub_8D1330(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v4; // r14
  __int64 v6; // r13
  bool v7; // si
  char v8; // al
  char v9; // cl
  unsigned __int8 v10; // di
  char v11; // al
  unsigned int v12; // r14d
  __int64 v14; // rax
  __int64 v15; // rax

  v4 = *a1;
  v6 = *a2;
  v7 = 0;
  v8 = *(_BYTE *)(*a1 + 140);
  v9 = *(_BYTE *)(v6 + 140);
  v10 = v8 - 9;
  if ( v8 == 2 )
    v7 = (*(_BYTE *)(v4 + 161) & 8) != 0;
  if ( v9 == 2 )
  {
    if ( !a3
      || !v7
      || (*(_BYTE *)(v6 + 161) & 8) == 0
      || dword_4F077C4 == 2
      && (*(_QWORD *)(v4 + 8) && (*(_BYTE *)(v4 + 162) & 8) == 0
       || *(_QWORD *)(v6 + 8) && (*(_BYTE *)(v6 + 162) & 8) == 0) )
    {
      goto LABEL_11;
    }
  }
  else
  {
    if ( !a3 || (unsigned __int8)(v9 - 9) > 2u )
      goto LABEL_11;
    if ( v10 > 2u )
      goto LABEL_12;
    if ( dword_4F077C4 == 2
      && (*(_QWORD *)(v4 + 8) && (*(_BYTE *)(v4 + 177) & 4) == 0
       || *(_QWORD *)(v6 + 8) && (*(_BYTE *)(v6 + 177) & 4) == 0) )
    {
LABEL_19:
      v14 = sub_8CA330(v4);
      if ( *a1 != v14 )
        goto LABEL_20;
LABEL_14:
      v11 = *(_BYTE *)(v6 + 140);
      v12 = 0;
      if ( (unsigned __int8)(v11 - 9) <= 2u )
        goto LABEL_21;
LABEL_15:
      if ( v11 == 2 )
      {
        if ( (*(_BYTE *)(v6 + 161) & 8) != 0 )
        {
          v15 = sub_8CA330(v6);
          if ( *a2 != v15 )
            goto LABEL_22;
        }
      }
      else if ( v11 == 12 )
      {
        if ( *(_QWORD *)(v6 + 8) )
        {
          v15 = sub_8CA330(v6);
          if ( *a2 != v15 )
            goto LABEL_22;
        }
      }
      return v12;
    }
  }
  if ( !(unsigned int)sub_8CD200((__int64 *)v4, v6) )
    sub_8CD200((__int64 *)v6, v4);
  v8 = *(_BYTE *)(v4 + 140);
  v10 = v8 - 9;
LABEL_11:
  if ( v10 <= 2u )
    goto LABEL_19;
LABEL_12:
  if ( v8 != 2 )
  {
    if ( v8 != 12 || !*(_QWORD *)(v4 + 8) )
      goto LABEL_14;
    goto LABEL_19;
  }
  if ( (*(_BYTE *)(v4 + 161) & 8) == 0 )
    goto LABEL_14;
  v14 = sub_8CA330(v4);
  if ( *a1 == v14 )
    goto LABEL_14;
LABEL_20:
  *a1 = v14;
  v11 = *(_BYTE *)(v6 + 140);
  v12 = 1;
  if ( (unsigned __int8)(v11 - 9) > 2u )
    goto LABEL_15;
LABEL_21:
  v15 = sub_8CA330(v6);
  if ( *a2 == v15 )
    return v12;
LABEL_22:
  *a2 = v15;
  return 1;
}
