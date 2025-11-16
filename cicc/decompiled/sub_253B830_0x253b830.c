// Function: sub_253B830
// Address: 0x253b830
//
__int64 __fastcall sub_253B830(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  bool v10; // cf
  bool v11; // zf
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
LABEL_19:
    if ( v3 == *(_QWORD *)(a1 + 24) )
      return 0;
    v13 = sub_220EF80(v3);
    v14 = *(_QWORD *)(v13 + 32);
    v3 = v13;
    v10 = v14 < *a2;
    v11 = v14 == *a2;
    if ( v14 == *a2 )
      goto LABEL_21;
LABEL_12:
    if ( v10 )
      return 0;
    if ( v11 )
    {
      v16 = a2[1];
      v15 = *(_QWORD *)(v3 + 40);
      goto LABEL_22;
    }
    return v3;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v3 + 32);
    if ( v4 != v7 )
    {
      if ( v4 < v7 )
        goto LABEL_4;
      if ( v4 != v7 )
        goto LABEL_9;
      v8 = a2[1];
      v9 = *(_QWORD *)(v3 + 40);
      goto LABEL_8;
    }
    v8 = a2[1];
    v9 = *(_QWORD *)(v3 + 40);
    if ( v8 == v9 )
      break;
LABEL_8:
    if ( v8 >= v9 )
      goto LABEL_9;
LABEL_4:
    v5 = *(_QWORD *)(v3 + 16);
    v6 = 1;
    if ( !v5 )
      goto LABEL_10;
LABEL_5:
    v3 = v5;
  }
  if ( *((_BYTE *)a2 + 16) < *(_BYTE *)(v3 + 48) )
    goto LABEL_4;
LABEL_9:
  v5 = *(_QWORD *)(v3 + 24);
  v6 = 0;
  if ( v5 )
    goto LABEL_5;
LABEL_10:
  if ( v6 )
    goto LABEL_19;
  v10 = v7 < v4;
  v11 = v7 == v4;
  if ( v7 != v4 )
    goto LABEL_12;
LABEL_21:
  v15 = *(_QWORD *)(v3 + 40);
  v16 = a2[1];
  if ( v15 == v16 )
  {
    if ( *(_BYTE *)(v3 + 48) < *((_BYTE *)a2 + 16) )
      return 0;
    return v3;
  }
LABEL_22:
  if ( v16 <= v15 )
    return v3;
  return 0;
}
