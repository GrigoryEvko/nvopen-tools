// Function: sub_214F160
// Address: 0x214f160
//
char __fastcall sub_214F160(__int64 a1, __int64 a2, const char *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // rbx
  bool v6; // zf
  char *v7; // rax
  bool v8; // r12
  __int64 v9; // rdi
  char v10; // dl
  _BYTE *v11; // rax
  char v12; // si
  __int64 v13; // rax
  char v14; // si

  v4 = a4;
  v5 = *(_QWORD *)(a2 + 24);
  if ( strcmp(a3, "vecelem") )
  {
    v6 = strcmp(a3, "vecv4comm1") == 0;
    LOBYTE(v7) = !v6;
    if ( v6 )
    {
      if ( (unsigned int)v5 <= 3 )
        return (char)v7;
      goto LABEL_18;
    }
    v6 = strcmp(a3, "vecv4comm2") == 0;
    LOBYTE(v7) = !v6;
    if ( v6 )
    {
      if ( (unsigned int)(v5 - 4) <= 3 )
        return (char)v7;
    }
    else
    {
      v6 = strcmp(a3, "vecv4pos") == 0;
      v8 = !v6;
      if ( v6 )
      {
        v13 = sub_1263B40(a4, "_");
        if ( (int)v5 >= 0 )
          v8 = v5;
        v9 = v13;
        v14 = a01230123[v8 & 3];
        v7 = *(char **)(v13 + 24);
        if ( (unsigned __int64)v7 < *(_QWORD *)(v9 + 16) )
        {
          *(_QWORD *)(v9 + 24) = v7 + 1;
          *v7 = v14;
          return (char)v7;
        }
        goto LABEL_29;
      }
      v6 = strcmp(a3, "vecv2comm1") == 0;
      LOBYTE(v7) = !v6;
      if ( !v6 )
      {
        v6 = strcmp(a3, "vecv2comm2") == 0;
        LOBYTE(v7) = !v6;
        if ( !v6 )
        {
          v9 = sub_1263B40(a4, "_");
          if ( (int)v5 < 0 )
            LOBYTE(v5) = 0;
          v10 = a01230123[v5 & 1];
          v7 = *(char **)(v9 + 24);
          if ( (unsigned __int64)v7 < *(_QWORD *)(v9 + 16) )
          {
            *(_QWORD *)(v9 + 24) = v7 + 1;
            *v7 = v10;
            return (char)v7;
          }
          v14 = a01230123[v5 & 1];
LABEL_29:
          LOBYTE(v7) = sub_16E7DE0(v9, v14);
          return (char)v7;
        }
        LODWORD(v5) = v5 - 2;
      }
      if ( (unsigned int)v5 <= 1 )
        return (char)v7;
    }
LABEL_18:
    LOBYTE(v7) = sub_1263B40(a4, "//");
    return (char)v7;
  }
  v11 = *(_BYTE **)(a4 + 24);
  if ( *(_BYTE **)(a4 + 16) == v11 )
  {
    v4 = sub_16E7EE0(a4, "_", 1u);
    v7 = *(char **)(v4 + 24);
  }
  else
  {
    *v11 = 95;
    v7 = (char *)(*(_QWORD *)(a4 + 24) + 1LL);
    *(_QWORD *)(a4 + 24) = v7;
  }
  v12 = a01230123[(int)v5];
  if ( *(_QWORD *)(v4 + 16) <= (unsigned __int64)v7 )
  {
    LOBYTE(v7) = sub_16E7DE0(v4, v12);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v7 + 1;
    *v7 = v12;
  }
  return (char)v7;
}
