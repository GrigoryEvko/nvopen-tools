// Function: sub_7ABD00
// Address: 0x7abd00
//
int __fastcall sub_7ABD00(__int64 ***a1, const char *a2, unsigned int a3)
{
  __int64 **v4; // rbx
  __int64 *v5; // rax
  __int64 *v6; // r14
  char *v7; // rax
  __int64 **v8; // rax
  __int64 **v9; // rbx
  char *v10; // rax
  __int64 *v11; // rax

  v4 = *a1;
  if ( *a1 )
  {
    while ( 1 )
    {
      LODWORD(v11) = strcmp((const char *)v4[1], a2);
      if ( !(_DWORD)v11 )
        break;
      if ( !*v4 )
      {
        v5 = (__int64 *)sub_822B10(16);
        *v5 = 0;
        v6 = v5;
        v5[1] = 0;
        v7 = (char *)sub_822B10(a3 + 1LL);
        v6[1] = (__int64)v7;
        strncpy(v7, a2, (int)a3);
        v11 = (__int64 *)v6[1];
        *((_BYTE *)v11 + (int)a3) = 0;
        *v4 = v6;
        return (int)v11;
      }
      v4 = (__int64 **)*v4;
    }
  }
  else
  {
    v8 = (__int64 **)sub_822B10(16);
    *v8 = 0;
    v9 = v8;
    v8[1] = 0;
    v10 = (char *)sub_822B10(a3 + 1LL);
    v9[1] = (__int64 *)v10;
    strncpy(v10, a2, (int)a3);
    v11 = v9[1];
    *((_BYTE *)v11 + (int)a3) = 0;
    *a1 = v9;
  }
  return (int)v11;
}
