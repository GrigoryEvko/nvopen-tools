// Function: sub_28513D0
// Address: 0x28513d0
//
__int64 __fastcall sub_28513D0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  bool v5; // al
  __int64 v6; // rax
  char v7; // si
  unsigned __int64 v8; // rdx
  char v9; // al
  bool v10; // cf
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  char v14; // dl

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
LABEL_18:
    if ( *(_QWORD *)(a1 + 24) == v3 )
      return 0;
    v12 = sub_220EF80(v3);
    v13 = *(_QWORD *)(v12 + 32);
    v3 = v12;
    v10 = v13 < *a2;
    if ( v13 == *a2 )
      goto LABEL_20;
LABEL_15:
    if ( v10 )
      return 0;
    return v3;
  }
  v4 = *a2;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v3 + 32);
    if ( v8 != v4 )
    {
      v5 = v8 > v4;
      goto LABEL_4;
    }
    v9 = *(_BYTE *)(v3 + 48);
    if ( !*((_BYTE *)a2 + 16) )
    {
      if ( v9 )
        goto LABEL_5;
      goto LABEL_10;
    }
    if ( !v9 )
      break;
LABEL_10:
    v5 = (__int64)a2[1] < *(_QWORD *)(v3 + 40);
LABEL_4:
    if ( !v5 )
      break;
LABEL_5:
    v6 = *(_QWORD *)(v3 + 16);
    v7 = 1;
    if ( !v6 )
      goto LABEL_13;
LABEL_6:
    v3 = v6;
  }
  v6 = *(_QWORD *)(v3 + 24);
  v7 = 0;
  if ( v6 )
    goto LABEL_6;
LABEL_13:
  if ( v7 )
    goto LABEL_18;
  v10 = v8 < v4;
  if ( v8 != v4 )
    goto LABEL_15;
LABEL_20:
  v14 = *((_BYTE *)a2 + 16);
  if ( *(_BYTE *)(v3 + 48) )
  {
    if ( v14 && *(_QWORD *)(v3 + 40) < (signed __int64)a2[1] )
      return 0;
    return v3;
  }
  if ( !v14 && *(_QWORD *)(v3 + 40) >= (signed __int64)a2[1] )
    return v3;
  return 0;
}
