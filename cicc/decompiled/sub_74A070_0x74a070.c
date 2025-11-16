// Function: sub_74A070
// Address: 0x74a070
//
void __fastcall sub_74A070(__int64 *a1, void (__fastcall **a2)(char *))
{
  __int64 *v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned __int8 v5; // al
  bool v6; // cc
  __int64 *v7; // rax

  if ( a1 )
  {
    v2 = a1;
    (*a2)("(");
    while ( 1 )
    {
      v5 = *((_BYTE *)v2 + 10);
      v6 = v5 <= 3u;
      if ( v5 == 3 )
        break;
LABEL_4:
      if ( !v6 )
      {
        if ( v5 != 4 )
          sub_721090();
        sub_74B930(v2[5], a2);
        goto LABEL_11;
      }
      if ( v5 )
      {
        ((void (__fastcall *)(__int64, void (__fastcall **)(char *)))*a2)(v2[5], a2);
        goto LABEL_11;
      }
      if ( !*v2 )
        goto LABEL_8;
LABEL_7:
      ((void (__fastcall *)(char *, void (__fastcall **)(char *)))*a2)(", ", a2);
      v2 = (__int64 *)*v2;
      if ( !v2 )
        goto LABEL_8;
    }
    while ( 1 )
    {
      sub_748000(v2[5], 0, (__int64)a2, v3, v4);
LABEL_11:
      v7 = (__int64 *)*v2;
      if ( !*v2 )
        break;
      if ( *((_BYTE *)v2 + 10) != 1 )
        goto LABEL_7;
      v2 = (__int64 *)*v2;
      v5 = *((_BYTE *)v7 + 10);
      v6 = v5 <= 3u;
      if ( v5 != 3 )
        goto LABEL_4;
    }
LABEL_8:
    ((void (__fastcall *)(char *, void (__fastcall **)(char *)))*a2)(")", a2);
  }
}
