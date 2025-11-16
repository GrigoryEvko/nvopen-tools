// Function: sub_8CA500
// Address: 0x8ca500
//
__int64 __fastcall sub_8CA500(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  char v5; // dl
  char v6; // dl
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdx

  v2 = a2;
  if ( !*(_QWORD *)(a2 + 32) )
    goto LABEL_22;
  while ( 1 )
  {
    sub_8CBB20(6, a1, v2);
    result = *(unsigned __int8 *)(a1 + 140);
    v5 = *(_BYTE *)(v2 + 140);
    if ( (_BYTE)result != v5 )
    {
      if ( (unsigned __int8)(result - 9) > 1u )
      {
        if ( (_BYTE)result == 11 )
          goto LABEL_32;
        if ( (_BYTE)result == 2 )
        {
          result = *(unsigned __int8 *)(a1 + 161);
          if ( (result & 8) == 0 || (**(_BYTE **)(a1 + 176) & 1) == 0 )
            return result;
          v10 = *(_QWORD *)(a1 + 168);
          if ( (result & 0x10) != 0 )
            v10 = *(_QWORD *)(v10 + 96);
          if ( !v10 )
            return result;
          do
          {
            sub_8C7090(2, v10);
            v10 = *(_QWORD *)(v10 + 120);
          }
          while ( v10 );
          result = *(unsigned __int8 *)(a1 + 140);
        }
        goto LABEL_6;
      }
      if ( (unsigned __int8)(v5 - 9) > 1u )
        goto LABEL_32;
    }
    v8 = *(_QWORD **)(v2 + 32);
    if ( !v8 )
      break;
    if ( a1 == *v8 )
      goto LABEL_38;
LABEL_26:
    if ( (unsigned __int8)(result - 9) <= 2u )
    {
      sub_8CAE10(a1);
      if ( (*(_BYTE *)(a1 + 177) & 0x10) == 0 )
        goto LABEL_30;
      v11 = qword_4F60248;
      if ( qword_4F60248 )
        qword_4F60248 = *(_QWORD *)qword_4F60248;
      else
        v11 = sub_823970(24);
      v12 = qword_4F60250;
      *(_BYTE *)(v11 + 8) = 6;
      qword_4F60250 = v11;
      *(_QWORD *)v11 = v12;
      *(_QWORD *)(v11 + 16) = a1;
      result = *(unsigned __int8 *)(a1 + 140);
    }
    else if ( (_BYTE)result == 2 )
    {
      if ( (*(_BYTE *)(a1 + 161) & 8) == 0 )
        return result;
      sub_8CA420(a1);
      goto LABEL_30;
    }
LABEL_6:
    if ( (_BYTE)result != 12 || !*(_QWORD *)(a1 + 8) )
      return result;
    do
    {
      a1 = *(_QWORD *)(a1 + 160);
      v6 = *(_BYTE *)(a1 + 140);
    }
    while ( v6 == 12 );
    for ( result = *(unsigned __int8 *)(v2 + 140); (_BYTE)result == 12; result = *(unsigned __int8 *)(v2 + 140) )
      v2 = *(_QWORD *)(v2 + 160);
    if ( (unsigned __int8)(v6 - 9) > 2u )
    {
      if ( v6 != 2
        || (*(_BYTE *)(a1 + 161) & 8) == 0
        || (*(_BYTE *)(a1 + 162) & 8) == 0
        || (_BYTE)result != 2
        || (*(_BYTE *)(v2 + 161) & 8) == 0
        || (*(_BYTE *)(v2 + 162) & 8) == 0 )
      {
        return result;
      }
    }
    else
    {
      if ( (*(_BYTE *)(a1 + 177) & 4) == 0 )
        return result;
      result = (unsigned int)(result - 9);
      if ( (unsigned __int8)result > 2u || (*(_BYTE *)(v2 + 177) & 4) == 0 )
        return result;
    }
    result = sub_8C7520((__int64 **)a1, (__int64 **)v2);
    if ( !(_DWORD)result )
      return result;
    if ( !*(_QWORD *)(v2 + 32) )
    {
LABEL_22:
      if ( *(_QWORD *)(a1 + 32) )
      {
        v7 = a1;
        a1 = v2;
        v2 = v7;
      }
    }
  }
  if ( a1 != v2 )
    goto LABEL_26;
LABEL_38:
  if ( (unsigned int)sub_8C6470(v2) )
  {
    v13 = *(_BYTE *)(v2 + 140);
    if ( (unsigned __int8)(v13 - 9) > 2u )
    {
      if ( v13 == 2 && (*(_BYTE *)(v2 + 161) & 8) != 0 )
      {
        sub_8CA420(v2);
        result = *(unsigned __int8 *)(a1 + 140);
        goto LABEL_6;
      }
      goto LABEL_30;
    }
    sub_8CAE10(v2);
    if ( (*(_BYTE *)(v2 + 177) & 0x10) == 0 )
      goto LABEL_30;
    v14 = qword_4F60248;
    if ( qword_4F60248 )
      qword_4F60248 = *(_QWORD *)qword_4F60248;
    else
      v14 = sub_823970(24);
    v15 = qword_4F60250;
    *(_BYTE *)(v14 + 8) = 6;
    qword_4F60250 = v14;
    *(_QWORD *)v14 = v15;
    *(_QWORD *)(v14 + 16) = v2;
    result = *(unsigned __int8 *)(a1 + 140);
    goto LABEL_6;
  }
  result = *(unsigned __int8 *)(a1 + 140);
  if ( (unsigned __int8)(result - 9) <= 2u )
  {
LABEL_32:
    if ( (unsigned int)sub_8D2490(a1) )
    {
      sub_8C9FB0(a1, 1u);
      result = *(unsigned __int8 *)(a1 + 140);
      goto LABEL_6;
    }
    goto LABEL_30;
  }
  if ( (_BYTE)result != 2 )
    goto LABEL_6;
  result = *(unsigned __int8 *)(a1 + 161);
  if ( (result & 8) != 0 && (**(_BYTE **)(a1 + 176) & 1) != 0 )
  {
    v9 = *(_QWORD *)(a1 + 168);
    if ( (result & 0x10) != 0 )
      v9 = *(_QWORD *)(v9 + 96);
    if ( v9 )
    {
      do
      {
        sub_8C7090(2, v9);
        v9 = *(_QWORD *)(v9 + 120);
      }
      while ( v9 );
LABEL_30:
      result = *(unsigned __int8 *)(a1 + 140);
      goto LABEL_6;
    }
  }
  return result;
}
