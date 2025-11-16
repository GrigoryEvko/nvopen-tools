// Function: sub_2A4DF00
// Address: 0x2a4df00
//
__int64 __fastcall sub_2A4DF00(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rax
  char v11; // al
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  char v14; // dl
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdx

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
LABEL_10:
    v8 = 0;
    if ( *(_QWORD *)(a1 + 24) != v3 )
    {
      v9 = sub_220EF80(v3);
      v4 = *a2;
      v7 = *(_QWORD *)(v9 + 32);
      v3 = v9;
      goto LABEL_12;
    }
    return v8;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v3 + 32);
    if ( v7 > v4 )
    {
LABEL_7:
      v5 = *(_QWORD *)(v3 + 16);
      v6 = 1;
      goto LABEL_8;
    }
    if ( v7 == v4 )
      break;
LABEL_4:
    v5 = *(_QWORD *)(v3 + 24);
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_5:
    v3 = v5;
  }
  v11 = *((_BYTE *)a2 + 24);
  if ( *(_BYTE *)(v3 + 56) )
  {
    if ( !v11 )
      goto LABEL_7;
    v12 = a2[1];
    v13 = *(_QWORD *)(v3 + 40);
    if ( v12 < v13 || v12 == v13 && a2[2] < *(_QWORD *)(v3 + 48) )
      goto LABEL_7;
    if ( v12 <= v13 && *(_QWORD *)(v3 + 48) >= a2[2] && a2[4] < *(_QWORD *)(v3 + 64) )
      goto LABEL_24;
    goto LABEL_4;
  }
  if ( v11 || a2[4] >= *(_QWORD *)(v3 + 64) )
    goto LABEL_4;
LABEL_24:
  v5 = *(_QWORD *)(v3 + 16);
  v6 = 1;
LABEL_8:
  if ( v5 )
    goto LABEL_5;
LABEL_9:
  if ( v6 )
    goto LABEL_10;
LABEL_12:
  if ( v7 < v4 )
    return 0;
  if ( v7 != v4 )
    return v3;
  v14 = *(_BYTE *)(v3 + 56);
  if ( !*((_BYTE *)a2 + 24) )
  {
    if ( !v14 )
      goto LABEL_34;
    return v3;
  }
  if ( !v14 )
    return 0;
  v15 = *(_QWORD *)(v3 + 40);
  v16 = a2[1];
  if ( v15 < v16 || v15 == v16 && *(_QWORD *)(v3 + 48) < a2[2] )
    return 0;
  if ( v15 > v16 || a2[2] < *(_QWORD *)(v3 + 48) )
    return v3;
LABEL_34:
  if ( *(_QWORD *)(v3 + 64) >= a2[4] )
    return v3;
  return 0;
}
