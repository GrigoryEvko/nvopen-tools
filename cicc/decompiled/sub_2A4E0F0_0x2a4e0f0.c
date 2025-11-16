// Function: sub_2A4E0F0
// Address: 0x2a4e0f0
//
__int64 __fastcall sub_2A4E0F0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  char v8; // r8
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r8
  char v11; // al
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax

  v3 = a1 + 8;
  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
    return v3;
  v5 = *a2;
  v6 = v3;
  do
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(v4 + 32) < v5 )
      {
LABEL_6:
        v4 = *(_QWORD *)(v4 + 24);
        goto LABEL_7;
      }
      if ( *(_QWORD *)(v4 + 32) == v5 )
        break;
LABEL_4:
      v6 = v4;
      v4 = *(_QWORD *)(v4 + 16);
      if ( !v4 )
        goto LABEL_8;
    }
    v8 = *(_BYTE *)(v4 + 56);
    if ( *((_BYTE *)a2 + 24) )
    {
      if ( !v8 )
        goto LABEL_6;
      v9 = *(_QWORD *)(v4 + 40);
      v10 = a2[1];
      if ( v9 < v10 || v9 == v10 && *(_QWORD *)(v4 + 48) < a2[2] )
        goto LABEL_6;
      if ( v9 > v10 || a2[2] < *(_QWORD *)(v4 + 48) || *(_QWORD *)(v4 + 64) >= a2[4] )
        goto LABEL_4;
    }
    else if ( v8 || *(_QWORD *)(v4 + 64) >= a2[4] )
    {
      goto LABEL_4;
    }
    v4 = *(_QWORD *)(v4 + 24);
LABEL_7:
    ;
  }
  while ( v4 );
LABEL_8:
  if ( v3 == v6 || *(_QWORD *)(v6 + 32) > v5 )
    return v3;
  if ( *(_QWORD *)(v6 + 32) == v5 )
  {
    v11 = *((_BYTE *)a2 + 24);
    if ( *(_BYTE *)(v6 + 56) )
    {
      if ( !v11 )
        return v3;
      v12 = a2[1];
      v13 = *(_QWORD *)(v6 + 40);
      if ( v12 < v13 || v12 == v13 && a2[2] < *(_QWORD *)(v6 + 48) )
        return v3;
      if ( v12 > v13 || *(_QWORD *)(v6 + 48) < a2[2] )
        return v6;
    }
    else if ( v11 )
    {
      return v6;
    }
    if ( a2[4] < *(_QWORD *)(v6 + 64) )
      return v3;
  }
  return v6;
}
