// Function: sub_31B7070
// Address: 0x31b7070
//
void __fastcall sub_31B7070(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rax

  v1 = a1[2];
  if ( !v1 )
    return;
  v2 = a1[3];
  while ( 1 )
  {
    if ( *(unsigned int *)(v1 + 8) <= v2
      || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2) + 144LL) != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2)
                                                                               + 16LL) )
    {
      return;
    }
    a1[3] = ++v2;
    if ( v2 < *(unsigned int *)(v1 + 8) )
      goto LABEL_33;
    v3 = a1[1];
    v4 = *a1;
    a1[3] = 0;
    a1[1] = v3 + 88;
    if ( v3 + 88 == *(_QWORD *)(v4 + 32) + 88LL * *(unsigned int *)(v4 + 40) )
      goto LABEL_37;
    v1 = v3 + 112;
    v2 = 0;
    a1[2] = v1;
    if ( *(_DWORD *)(v1 + 8) )
      break;
LABEL_35:
    v1 = a1[2];
    if ( !v1 )
      return;
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2) + 144LL) != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2)
                                                                               + 16LL) )
      goto LABEL_35;
    a1[3] = ++v2;
    if ( v2 < *(unsigned int *)(v1 + 8) )
      goto LABEL_30;
    v5 = a1[1];
    v6 = *a1;
    a1[3] = 0;
    a1[1] = v5 + 88;
    if ( v5 + 88 == *(_QWORD *)(v6 + 32) + 88LL * *(unsigned int *)(v6 + 40) )
      goto LABEL_37;
    v7 = v5 + 112;
    v2 = 0;
    a1[2] = v7;
    if ( *(_DWORD *)(v7 + 8) )
      break;
LABEL_32:
    v1 = a1[2];
LABEL_33:
    if ( !v1 )
      return;
    if ( *(unsigned int *)(v1 + 8) <= v2 )
      goto LABEL_35;
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v7 + 8 * v2) + 144LL) != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v7 + 8 * v2)
                                                                               + 16LL) )
      goto LABEL_32;
    a1[3] = ++v2;
    if ( v2 < *(unsigned int *)(v7 + 8) )
      goto LABEL_28;
    v8 = a1[1];
    v9 = *a1;
    a1[3] = 0;
    a1[1] = v8 + 88;
    if ( v8 + 88 == *(_QWORD *)(v9 + 32) + 88LL * *(unsigned int *)(v9 + 40) )
      goto LABEL_37;
    v10 = v8 + 112;
    v2 = 0;
    a1[2] = v10;
    if ( *(_DWORD *)(v10 + 8) )
      break;
LABEL_30:
    v7 = a1[2];
    if ( !v7 )
      return;
    if ( *(unsigned int *)(v7 + 8) <= v2 )
      goto LABEL_32;
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 8 * v2) + 144LL) != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 8 * v2)
                                                                                + 16LL) )
      goto LABEL_30;
    a1[3] = ++v2;
    if ( v2 < *(unsigned int *)(v10 + 8) )
      break;
    v11 = a1[1];
    v12 = *a1;
    a1[3] = 0;
    a1[1] = v11 + 88;
    if ( v11 + 88 == *(_QWORD *)(v12 + 32) + 88LL * *(unsigned int *)(v12 + 40) )
    {
LABEL_37:
      a1[2] = 0;
      return;
    }
    v13 = v11 + 112;
    v2 = 0;
    a1[2] = v13;
LABEL_27:
    while ( *(unsigned int *)(v13 + 8) > v2
         && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 8 * v2) + 144LL) == *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 8 * v2)
                                                                                   + 16LL) )
    {
      v14 = v2 + 1;
      a1[3] = v14;
      if ( v14 >= *(unsigned int *)(v13 + 8) )
      {
        v15 = a1[1];
        v16 = *a1;
        a1[3] = 0;
        v17 = v15 + 88;
        v18 = v15 + 112;
        a1[1] = v17;
        if ( v17 == *(_QWORD *)(v16 + 32) + 88LL * *(unsigned int *)(v16 + 40) )
          v18 = 0;
        a1[2] = v18;
      }
      sub_31B7070(a1);
      v13 = a1[2];
      if ( !v13 )
        return;
      v2 = a1[3];
    }
LABEL_28:
    v10 = a1[2];
    if ( !v10 )
      return;
    if ( *(unsigned int *)(v10 + 8) <= v2 )
      goto LABEL_30;
  }
  v13 = a1[2];
  if ( v13 )
    goto LABEL_27;
}
