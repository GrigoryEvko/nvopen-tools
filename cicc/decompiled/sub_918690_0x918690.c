// Function: sub_918690
// Address: 0x918690
//
__int64 __fastcall sub_918690(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // al
  unsigned __int8 v13; // dl

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_9185D0((__int64)a1, (unsigned __int64 *)a3);
    v10 = a1[4];
    v11 = *(_QWORD *)(v10 + 32);
    if ( *(_QWORD *)a3 <= v11 && (*(_QWORD *)a3 != v11 || *(_BYTE *)(a3 + 8) <= *(_BYTE *)(v10 + 40)) )
      return sub_9185D0((__int64)a1, (unsigned __int64 *)a3);
    return 0;
  }
  v4 = *(_QWORD *)a3;
  if ( *(_QWORD *)(a2 + 32) > *(_QWORD *)a3 )
    goto LABEL_3;
  if ( *(_QWORD *)(a2 + 32) == *(_QWORD *)a3 )
  {
    v12 = *(_BYTE *)(a3 + 8);
    v13 = *(_BYTE *)(a2 + 40);
    if ( v13 > v12 )
    {
LABEL_3:
      result = a2;
      if ( a1[3] == a2 )
        return result;
      v6 = sub_220EF80(a2);
      v7 = v6;
      if ( v4 > *(_QWORD *)(v6 + 32) || v4 == *(_QWORD *)(v6 + 32) && *(_BYTE *)(v6 + 40) < *(_BYTE *)(a3 + 8) )
      {
        result = 0;
        if ( *(_QWORD *)(v7 + 24) )
          return a2;
        return result;
      }
      return sub_9185D0((__int64)a1, (unsigned __int64 *)a3);
    }
  }
  else
  {
    if ( *(_QWORD *)(a2 + 32) < *(_QWORD *)a3 )
      goto LABEL_12;
    v13 = *(_BYTE *)(a2 + 40);
    v12 = *(_BYTE *)(a3 + 8);
  }
  if ( v13 >= v12 )
    return a2;
LABEL_12:
  if ( a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  v9 = v8;
  if ( v4 >= *(_QWORD *)(v8 + 32) && (v4 != *(_QWORD *)(v8 + 32) || *(_BYTE *)(a3 + 8) >= *(_BYTE *)(v8 + 40)) )
    return sub_9185D0((__int64)a1, (unsigned __int64 *)a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v9;
  return result;
}
