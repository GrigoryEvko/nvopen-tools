// Function: sub_2CFDCE0
// Address: 0x2cfdce0
//
char __fastcall sub_2CFDCE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  size_t v7; // rdx
  char **v8; // rax
  char *v9; // r15
  __int64 v10; // r14
  size_t v11; // rdx
  __int64 v12; // r15
  size_t v13; // rdx
  char *v14; // r15
  __int64 i; // rbx
  _BYTE *v16; // rsi
  unsigned __int8 v17; // r12
  size_t v18; // rdx
  char *v19; // r14
  __int64 j; // rbx
  _BYTE *v21; // rsi
  __int64 v23; // [rsp+8h] [rbp-48h]

  v7 = 0;
  v23 = *(_QWORD *)(a4 + 72);
  v8 = off_4C5D0D8;
  v9 = off_4C5D0D8[0];
  if ( off_4C5D0D8[0] )
  {
    v8 = (char **)strlen(off_4C5D0D8[0]);
    v7 = (size_t)v8;
  }
  v10 = *(_QWORD *)(a2 + 48);
  if ( v10 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v10 = sub_B91F50(a2, v9, v7);
    LOBYTE(v8) = *(_BYTE *)a3;
    if ( *(_BYTE *)a3 <= 0x1Cu )
    {
      if ( v10 )
      {
        if ( (_BYTE)v8 == 22 )
        {
          v8 = *(char ***)(a3 + 8);
          if ( *((_BYTE *)v8 + 8) == 14 )
          {
            v8 = *(char ***)(a3 + 16);
            if ( v8 )
            {
              if ( !v8[1] )
                LOBYTE(v8) = sub_B2D400(a3, 22);
            }
          }
        }
      }
      return (char)v8;
    }
    v8 = off_4C5D0D8;
    v9 = off_4C5D0D8[0];
  }
  else if ( *(_BYTE *)a3 <= 0x1Cu )
  {
    return (char)v8;
  }
  v11 = 0;
  if ( v9 )
  {
    v8 = (char **)strlen(v9);
    v11 = (size_t)v8;
  }
  if ( !*(_QWORD *)(a3 + 48) && (*(_BYTE *)(a3 + 7) & 0x20) == 0 )
  {
    if ( !v10 )
      return (char)v8;
LABEL_15:
    if ( (unsigned __int8)(*(_BYTE *)a3 - 78) > 1u || (v8 = *(char ***)(a3 + 16)) != 0 && !v8[1] )
    {
      v13 = 0;
      v14 = off_4C5D0D8[0];
      if ( off_4C5D0D8[0] )
        v13 = strlen(off_4C5D0D8[0]);
      sub_B9A090(a3, v14, v13, v10);
      sub_CEF900(2, v23);
      LOBYTE(v8) = sub_2CFD5A0(a1, a3, a4, a5);
      for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v16 = *(_BYTE **)(i + 24);
        if ( *v16 > 0x1Cu )
          LOBYTE(v8) = sub_2CFD5A0(a1, (__int64)v16, a4, a5);
      }
    }
    return (char)v8;
  }
  v12 = sub_B91F50(a3, v9, v11);
  LOBYTE(v8) = v12 == 0;
  if ( (v10 != 0) != (v12 == 0) )
    return (char)v8;
  if ( v10 )
  {
    if ( v12 )
      return (char)v8;
    goto LABEL_15;
  }
  if ( *(_BYTE *)a2 == 61 )
  {
    if ( a5 && !(_BYTE)qword_5014B28 )
      return (char)v8;
    v17 = 1;
  }
  else
  {
    v17 = a5;
  }
  v18 = 0;
  v19 = off_4C5D0D8[0];
  if ( off_4C5D0D8[0] )
    v18 = strlen(off_4C5D0D8[0]);
  sub_B9A090(a2, v19, v18, v12);
  sub_CEF900(2, v23);
  LOBYTE(v8) = sub_2CFD5A0(a1, a2, a4, v17);
  for ( j = *(_QWORD *)(a2 + 16); j; j = *(_QWORD *)(j + 8) )
  {
    v21 = *(_BYTE **)(j + 24);
    if ( *v21 > 0x1Cu )
      LOBYTE(v8) = sub_2CFD5A0(a1, (__int64)v21, a4, v17);
  }
  return (char)v8;
}
