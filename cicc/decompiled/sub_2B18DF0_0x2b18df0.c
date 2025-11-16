// Function: sub_2B18DF0
// Address: 0x2b18df0
//
__int64 __fastcall sub_2B18DF0(char *a1, char *a2)
{
  char *v2; // r12
  char *v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rax
  char *v7; // rax
  char *v8; // rax
  bool v10; // zf
  int v11; // [rsp+8h] [rbp-48h]
  int v12; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+18h] [rbp-38h]

  v12 = sub_2B18C70(a1, 0);
  v11 = sub_2B18C70(a2, 0);
  if ( a1 == a2 )
    return 1;
  v2 = a1;
  v3 = a2;
  do
  {
    if ( v2
      && (a1 == v2 || (v6 = *((_QWORD *)v2 + 2)) != 0 && !*(_QWORD *)(v6 + 8))
      && (v13 = sub_2B18C70(v2, 0), v4 = HIDWORD(v13), BYTE4(v13))
      && v11 != (_DWORD)v13 )
    {
      v7 = (char *)*((_QWORD *)v2 - 12);
      if ( *v7 == 91 )
      {
        v10 = v7 == v2;
        v2 = (char *)*((_QWORD *)v2 - 12);
        LOBYTE(v4) = v10;
      }
      else
      {
        v2 = 0;
      }
    }
    else
    {
      LOBYTE(v4) = 1;
    }
    if ( !v3 )
      goto LABEL_7;
    if ( a2 != v3 )
    {
      v5 = *((_QWORD *)v3 + 2);
      if ( !v5 || *(_QWORD *)(v5 + 8) )
        goto LABEL_7;
    }
    v14 = sub_2B18C70(v3, 0);
    if ( !BYTE4(v14) || v12 == (_DWORD)v14 )
      goto LABEL_7;
    v8 = (char *)*((_QWORD *)v3 - 12);
    if ( *v8 != 91 )
    {
      v3 = 0;
LABEL_7:
      if ( (_BYTE)v4 )
        goto LABEL_29;
      goto LABEL_8;
    }
    if ( (_BYTE)v4 && v8 == v3 )
LABEL_29:
      BUG();
    v3 = (char *)*((_QWORD *)v3 - 12);
LABEL_8:
    if ( a1 == v3 )
      return 1;
  }
  while ( a2 != v2 );
  return 0;
}
