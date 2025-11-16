// Function: sub_2A513C0
// Address: 0x2a513c0
//
__int64 __fastcall sub_2A513C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  char v10; // al
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  char v13; // al
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  char v17; // al
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    v9 = a1[4];
    if ( *(_QWORD *)(v9 + 32) < *(_QWORD *)a3 )
      return 0;
    if ( *(_QWORD *)(v9 + 32) != *(_QWORD *)a3 )
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    v10 = *(_BYTE *)(v9 + 56);
    if ( *(_BYTE *)(a3 + 24) )
    {
      if ( !v10 )
        return 0;
      v11 = *(_QWORD *)(v9 + 40);
      v12 = *(_QWORD *)(a3 + 8);
      if ( v11 < v12 || v11 == v12 && *(_QWORD *)(v9 + 48) < *(_QWORD *)(a3 + 16) )
        return 0;
      if ( v11 > v12 || *(_QWORD *)(a3 + 16) < *(_QWORD *)(v9 + 48) )
        return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    }
    else if ( v10 )
    {
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    }
    if ( *(_QWORD *)(v9 + 64) >= *(_QWORD *)(a3 + 32) )
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    return 0;
  }
  v4 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 < *(_QWORD *)(a2 + 32) )
    goto LABEL_3;
  if ( *(_QWORD *)a3 == *(_QWORD *)(a2 + 32) )
  {
    v13 = *(_BYTE *)(a3 + 24);
    if ( *(_BYTE *)(a2 + 56) )
    {
      if ( !v13 )
        goto LABEL_3;
      v20 = *(_QWORD *)(a3 + 8);
      v21 = *(_QWORD *)(a2 + 40);
      if ( v20 < v21 || v20 == v21 && *(_QWORD *)(a3 + 16) < *(_QWORD *)(a2 + 48) )
        goto LABEL_3;
      if ( v20 > v21 || *(_QWORD *)(a2 + 48) < *(_QWORD *)(a3 + 16) )
        goto LABEL_31;
    }
    else if ( v13 )
    {
      goto LABEL_31;
    }
    if ( *(_QWORD *)(a3 + 32) >= *(_QWORD *)(a2 + 64) )
      goto LABEL_31;
LABEL_3:
    if ( a1[3] == a2 )
      return a2;
    v5 = sub_220EF80(a2);
    v6 = (_QWORD *)v5;
    if ( v4 > *(_QWORD *)(v5 + 32) )
      goto LABEL_48;
    if ( v4 != *(_QWORD *)(v5 + 32) )
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    v17 = *(_BYTE *)(v5 + 56);
    if ( *(_BYTE *)(a3 + 24) )
    {
      if ( !v17 || (v18 = v6[5], v19 = *(_QWORD *)(a3 + 8), v18 < v19) || v18 == v19 && v6[6] < *(_QWORD *)(a3 + 16) )
      {
LABEL_48:
        result = 0;
        if ( v6[3] )
          return a2;
        return result;
      }
      if ( v18 > v19 || *(_QWORD *)(a3 + 16) < v6[6] )
        return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    }
    else if ( v17 )
    {
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    }
    if ( v6[8] >= *(_QWORD *)(a3 + 32) )
      return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
    goto LABEL_48;
  }
  if ( *(_QWORD *)a3 > *(_QWORD *)(a2 + 32) )
    goto LABEL_9;
LABEL_31:
  v14 = *(_BYTE *)(a2 + 56);
  if ( *(_BYTE *)(a3 + 24) )
  {
    if ( !v14 )
      goto LABEL_9;
    v15 = *(_QWORD *)(a2 + 40);
    v16 = *(_QWORD *)(a3 + 8);
    if ( v15 < v16 || v15 == v16 && *(_QWORD *)(a2 + 48) < *(_QWORD *)(a3 + 16) )
      goto LABEL_9;
    if ( v15 > v16 || *(_QWORD *)(a3 + 16) < *(_QWORD *)(a2 + 48) )
      return a2;
  }
  else if ( v14 )
  {
    return a2;
  }
  if ( *(_QWORD *)(a2 + 64) >= *(_QWORD *)(a3 + 32) )
    return a2;
LABEL_9:
  if ( a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  if ( !sub_2A4D650(a3, v8 + 32) )
    return sub_2A4DF00((__int64)a1, (unsigned __int64 *)a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v8;
  return result;
}
