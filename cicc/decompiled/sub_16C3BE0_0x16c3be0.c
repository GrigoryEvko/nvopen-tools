// Function: sub_16C3BE0
// Address: 0x16c3be0
//
__int64 __fastcall sub_16C3BE0(char *a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v4; // rcx
  __int64 v5; // rbx
  char *v7; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v8; // [rsp+8h] [rbp-18h]

  v7 = a1;
  v8 = a2;
  if ( a2 )
  {
    if ( sub_16C36C0(a1[a2 - 1], a3) )
      return v8 - 1;
    a2 = v8;
  }
  v4 = a2 - 1;
  if ( !a3 )
  {
    v5 = sub_16D25A0(&v7, "\\/", 2, v4);
    if ( v5 != -1 )
      goto LABEL_4;
    v5 = v8 - 2;
    if ( v8 < 2 )
      v5 = v8;
    while ( v5 )
    {
      if ( v7[--v5] == 58 )
        goto LABEL_4;
    }
    return 0;
  }
  v5 = sub_16D25A0(&v7, "/", 1, v4);
  if ( v5 == -1 )
    return 0;
LABEL_4:
  if ( v5 == 1 && sub_16C36C0(*v7, a3) )
    return 0;
  else
    return v5 + 1;
}
