// Function: sub_18DBA00
// Address: 0x18dba00
//
void *__fastcall sub_18DBA00(char *a1, _BYTE *a2, char a3)
{
  int v5; // ecx
  int v6; // eax
  char v7; // dl
  void *result; // rax

  v5 = (unsigned __int8)a2[2];
  v6 = (unsigned __int8)a1[2];
  v7 = *a1;
  if ( v6 == v5 )
  {
    a1[2] = v5;
    *a1 = *a2 & v7;
    if ( v6 )
      goto LABEL_14;
LABEL_20:
    a1[1] = 0;
    return sub_18DB490((__int64)(a1 + 8));
  }
  if ( !a1[2] || !a2[2] )
    goto LABEL_19;
  if ( (unsigned __int8)v5 < (unsigned __int8)v6 )
  {
    v6 = (unsigned __int8)a2[2];
    v5 = (unsigned __int8)a1[2];
  }
  if ( a3 )
  {
    if ( (unsigned int)(v6 - 1) <= 1 && (unsigned int)(v5 - 2) <= 1 )
    {
      LOBYTE(v6) = v5;
      goto LABEL_12;
    }
    goto LABEL_19;
  }
  if ( (unsigned int)(v6 - 2) <= 1 )
  {
    if ( (unsigned int)(v5 - 3) <= 3 )
      goto LABEL_12;
    goto LABEL_19;
  }
  if ( v6 == 4 )
  {
    if ( (unsigned int)(v5 - 5) <= 1 )
    {
      LOBYTE(v6) = 4;
      goto LABEL_12;
    }
    goto LABEL_19;
  }
  if ( v5 != 6 || v6 != 5 )
  {
LABEL_19:
    a1[2] = 0;
    *a1 = *a2 & v7;
    goto LABEL_20;
  }
  LOBYTE(v6) = 5;
LABEL_12:
  a1[2] = v6;
  *a1 = *a2 & v7;
LABEL_14:
  if ( a1[1] || a2[1] )
    return sub_18DB9D0((__int64)a1, 0);
  result = (void *)sub_18DB560((__int64)(a1 + 8), (__int64)(a2 + 8));
  a1[1] = (char)result;
  return result;
}
