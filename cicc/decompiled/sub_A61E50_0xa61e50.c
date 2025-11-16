// Function: sub_A61E50
// Address: 0xa61e50
//
void __fastcall sub_A61E50(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  void (*v4)(); // rax
  const char *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  if ( *(_BYTE *)a2 == 85 )
  {
    v7 = *(_QWORD *)(a2 - 32);
    if ( v7 )
    {
      if ( !*(_BYTE *)v7
        && *(_QWORD *)(v7 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v7 + 33) & 0x20) != 0
        && *(_DWORD *)(v7 + 36) == 149 )
      {
        sub_904010(*a1, " ; (");
        v8 = sub_B5B740(a2);
        sub_A5B360(a1, v8, 0);
        sub_904010(*a1, ", ");
        v9 = sub_B5B890(a2);
        sub_A5B360(a1, v9, 0);
        sub_904010(*a1, ")");
      }
    }
  }
  v3 = a1[33];
  if ( v3 )
  {
    v4 = *(void (**)())(*(_QWORD *)v3 + 48LL);
    if ( v4 != nullsub_34 )
      ((void (__fastcall *)(__int64, __int64, __int64))v4)(v3, a2, *a1);
  }
  if ( (_BYTE)qword_4F80AA8 )
  {
    if ( *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_14;
    if ( !*(_QWORD *)(a2 + 48) )
    {
      if ( !(_BYTE)qword_4F809C8 )
        goto LABEL_14;
      goto LABEL_11;
    }
    sub_904010(*a1, " ; ");
    sub_B10EE0(a2 + 48, *a1);
  }
  if ( !(_BYTE)qword_4F809C8 || *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_14;
LABEL_11:
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v5 = (const char *)sub_B91C10(a2, 2);
    if ( v5 )
    {
      sub_904010(*a1, " ; ");
      sub_A61DE0(v5, *a1, a1[1]);
    }
  }
LABEL_14:
  if ( (_BYTE)qword_4F80B88 )
  {
    v6 = sub_904010(*a1, " ; ");
    sub_CB5A80(v6, a2);
  }
}
