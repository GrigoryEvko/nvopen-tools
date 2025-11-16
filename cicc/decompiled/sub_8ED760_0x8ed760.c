// Function: sub_8ED760
// Address: 0x8ed760
//
char *__fastcall sub_8ED760(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 v3; // al
  unsigned __int8 *v5; // r14
  unsigned __int8 *v6; // rax
  unsigned __int8 *v7; // rdi
  char *v8; // rbx
  __int64 v9; // rax
  unsigned __int8 *v10; // rdi
  unsigned __int8 v11; // al
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rdi

  v2 = a1;
  v3 = *a1;
  if ( *a1 == 103 )
  {
    if ( a1[1] != 115 )
      return sub_8ECD10(v2, a2);
    if ( !*(_QWORD *)(a2 + 32) )
      sub_8E5790((unsigned __int8 *)"::", a2);
    v3 = a1[2];
    v2 = a1 + 2;
  }
  if ( v3 != 115 || v2[1] != 114 )
    return sub_8ECD10(v2, a2);
  v5 = v2 + 2;
  LODWORD(v6) = v2[2];
  if ( (unsigned int)((_DWORD)v6 - 48) > 9 )
  {
    if ( dword_4D0425C )
    {
      ++*(_QWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 48);
      v8 = sub_8E9FF0((__int64)(v2 + 2), 0, 0, 0, 1u, a2);
      sub_8EB260(v2 + 2, 0, 0, a2);
      v9 = *(_QWORD *)(a2 + 32);
      --*(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 32) = v9 - 1;
      if ( *v8 == 78 )
      {
        *(_QWORD *)(a2 + 32) = v9;
        v13 = v2 + 2;
        v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(v2 + 2), 0, 0, 0, 1u, a2);
        sub_8EB260(v13, 0, 0, a2);
        --*(_QWORD *)(a2 + 32);
        goto LABEL_28;
      }
      LOBYTE(v6) = v2[2];
    }
    if ( (_BYTE)v6 != 78 )
    {
      v7 = v2 + 2;
      v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(v2 + 2), 0, 0, 0, 1u, a2);
      sub_8EB260(v7, 0, 0, a2);
      if ( !*(_QWORD *)(a2 + 32) )
        sub_8E5790((unsigned __int8 *)"::", a2);
      goto LABEL_28;
    }
    v10 = v2 + 3;
    v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(v2 + 3), 0, 0, 0, 1u, a2);
    sub_8EB260(v10, 0, 0, a2);
    if ( !*(_QWORD *)(a2 + 32) )
      sub_8E5790((unsigned __int8 *)"::", a2);
    v11 = *v2;
    if ( *(_DWORD *)(a2 + 24) )
    {
LABEL_41:
      if ( *v2 != 69 )
        return (char *)v2;
LABEL_42:
      ++v2;
LABEL_28:
      if ( *(_DWORD *)(a2 + 24) )
        return (char *)v2;
      return sub_8ECD10(v2, a2);
    }
    while ( 1 )
    {
      if ( v11 == 69 )
        goto LABEL_42;
      if ( !v11 )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
        goto LABEL_41;
      }
      v12 = sub_8E72C0(v2, 0, a2);
      v2 = v12;
      if ( *(_DWORD *)(a2 + 24) )
        break;
      v11 = *v12;
      if ( v11 == 73 )
      {
        v2 = (unsigned __int8 *)sub_8E9020(v2, a2);
        if ( *(_QWORD *)(a2 + 32) )
          goto LABEL_47;
        goto LABEL_46;
      }
      if ( !*(_QWORD *)(a2 + 32) )
      {
LABEL_46:
        sub_8E5790((unsigned __int8 *)"::", a2);
LABEL_47:
        v11 = *v2;
        if ( *(_DWORD *)(a2 + 24) )
          goto LABEL_41;
      }
    }
    if ( *(_QWORD *)(a2 + 32) )
      goto LABEL_41;
    goto LABEL_46;
  }
  if ( !*(_DWORD *)(a2 + 24) )
  {
    while ( 1 )
    {
      if ( (_BYTE)v6 == 69 )
        goto LABEL_30;
      if ( !(_BYTE)v6 )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
        LOBYTE(v6) = *v5;
        goto LABEL_14;
      }
      v6 = sub_8E72C0(v5, 0, a2);
      v5 = v6;
      if ( *(_DWORD *)(a2 + 24) )
        break;
      LOBYTE(v6) = *v6;
      if ( (_BYTE)v6 == 73 )
      {
        v5 = (unsigned __int8 *)sub_8E9020(v5, a2);
        if ( *(_QWORD *)(a2 + 32) )
          goto LABEL_20;
        goto LABEL_19;
      }
      if ( !*(_QWORD *)(a2 + 32) )
      {
LABEL_19:
        sub_8E5790((unsigned __int8 *)"::", a2);
LABEL_20:
        LOBYTE(v6) = *v5;
        if ( *(_DWORD *)(a2 + 24) )
          goto LABEL_14;
      }
    }
    if ( *(_QWORD *)(a2 + 32) )
    {
      LOBYTE(v6) = *v6;
      goto LABEL_14;
    }
    goto LABEL_19;
  }
LABEL_14:
  v2 = v5;
  if ( (_BYTE)v6 == 69 )
  {
LABEL_30:
    v2 = v5 + 1;
    goto LABEL_28;
  }
  return (char *)v2;
}
