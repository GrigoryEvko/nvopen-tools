// Function: sub_F1BF80
// Address: 0xf1bf80
//
__int64 __fastcall sub_F1BF80(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  char v10; // al
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  char v17; // al
  char v18; // al
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  char v21; // al
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    v13 = a1[4];
    if ( *(_QWORD *)(v13 + 32) < *(_QWORD *)a3 )
      return 0;
    if ( *(_QWORD *)(v13 + 32) != *(_QWORD *)a3 )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    v14 = *(_BYTE *)(v13 + 56);
    if ( *(_BYTE *)(a3 + 24) )
    {
      if ( !v14 )
        return 0;
      v15 = *(_QWORD *)(v13 + 40);
      v16 = *(_QWORD *)(a3 + 8);
      if ( v15 < v16 || v15 == v16 && *(_QWORD *)(v13 + 48) < *(_QWORD *)(a3 + 16) )
        return 0;
      if ( v15 > v16 || *(_QWORD *)(a3 + 16) < *(_QWORD *)(v13 + 48) )
        return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    }
    else if ( v14 )
    {
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    }
    if ( *(_QWORD *)(v13 + 64) >= *(_QWORD *)(a3 + 32) )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    return 0;
  }
  v4 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 < *(_QWORD *)(a2 + 32) )
    goto LABEL_3;
  if ( *(_QWORD *)a3 == *(_QWORD *)(a2 + 32) )
  {
    v17 = *(_BYTE *)(a3 + 24);
    if ( *(_BYTE *)(a2 + 56) )
    {
      if ( !v17 )
        goto LABEL_3;
      v24 = *(_QWORD *)(a3 + 8);
      v25 = *(_QWORD *)(a2 + 40);
      if ( v24 < v25 || v24 == v25 && *(_QWORD *)(a3 + 16) < *(_QWORD *)(a2 + 48) )
        goto LABEL_3;
      if ( v24 > v25 || *(_QWORD *)(a2 + 48) < *(_QWORD *)(a3 + 16) )
        goto LABEL_40;
    }
    else if ( v17 )
    {
      goto LABEL_40;
    }
    if ( *(_QWORD *)(a3 + 32) >= *(_QWORD *)(a2 + 64) )
      goto LABEL_40;
LABEL_3:
    if ( a1[3] == a2 )
      return a2;
    v5 = sub_220EF80(a2);
    v6 = (_QWORD *)v5;
    if ( v4 > *(_QWORD *)(v5 + 32) )
      goto LABEL_57;
    if ( v4 != *(_QWORD *)(v5 + 32) )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    v21 = *(_BYTE *)(v5 + 56);
    if ( *(_BYTE *)(a3 + 24) )
    {
      if ( !v21 || (v22 = v6[5], v23 = *(_QWORD *)(a3 + 8), v22 < v23) || v22 == v23 && v6[6] < *(_QWORD *)(a3 + 16) )
      {
LABEL_57:
        result = 0;
        if ( v6[3] )
          return a2;
        return result;
      }
      if ( v22 > v23 || *(_QWORD *)(a3 + 16) < v6[6] )
        return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    }
    else if ( v21 )
    {
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    }
    if ( v6[8] >= *(_QWORD *)(a3 + 32) )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    goto LABEL_57;
  }
  if ( *(_QWORD *)a3 > *(_QWORD *)(a2 + 32) )
    goto LABEL_9;
LABEL_40:
  v18 = *(_BYTE *)(a2 + 56);
  if ( *(_BYTE *)(a3 + 24) )
  {
    if ( !v18 )
      goto LABEL_9;
    v19 = *(_QWORD *)(a2 + 40);
    v20 = *(_QWORD *)(a3 + 8);
    if ( v19 < v20 || v19 == v20 && *(_QWORD *)(a2 + 48) < *(_QWORD *)(a3 + 16) )
      goto LABEL_9;
    if ( v19 > v20 || *(_QWORD *)(a3 + 16) < *(_QWORD *)(a2 + 48) )
      return a2;
  }
  else if ( v18 )
  {
    return a2;
  }
  if ( *(_QWORD *)(a2 + 64) >= *(_QWORD *)(a3 + 32) )
    return a2;
LABEL_9:
  if ( a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  v9 = v8;
  if ( v4 >= *(_QWORD *)(v8 + 32) )
  {
    if ( v4 != *(_QWORD *)(v8 + 32) )
      return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
    v10 = *(_BYTE *)(a3 + 24);
    if ( !*(_BYTE *)(v9 + 56) )
    {
      if ( v10 )
        return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
LABEL_19:
      if ( *(_QWORD *)(a3 + 32) >= *(_QWORD *)(v9 + 64) )
        return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
      goto LABEL_20;
    }
    if ( v10 )
    {
      v11 = *(_QWORD *)(a3 + 8);
      v12 = *(_QWORD *)(v9 + 40);
      if ( v11 >= v12 && (v11 != v12 || *(_QWORD *)(a3 + 16) >= *(_QWORD *)(v9 + 48)) )
      {
        if ( v11 > v12 || *(_QWORD *)(v9 + 48) < *(_QWORD *)(a3 + 16) )
          return sub_F18D30((__int64)a1, (unsigned __int64 *)a3);
        goto LABEL_19;
      }
    }
  }
LABEL_20:
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v9;
  return result;
}
