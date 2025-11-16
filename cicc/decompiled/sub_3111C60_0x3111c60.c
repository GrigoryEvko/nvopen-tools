// Function: sub_3111C60
// Address: 0x3111c60
//
__int64 __fastcall sub_3111C60(__int64 a1, int a2, int a3, char a4)
{
  __int64 v4; // rbx
  char *v5; // r13
  unsigned __int64 v6; // rdx
  char *v7; // r13
  size_t v8; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( a3 == 5 && a4 )
  {
    v4 = a2;
    v7 = off_49D8BE0[a2];
    v8 = strlen(v7);
    sub_2241130((unsigned __int64 *)a1, 0, 0, v7, v8);
  }
  else
  {
    v4 = a2;
    if ( a3 == 1 )
    {
      v5 = off_49D8BF0[a2];
      v6 = strlen(v5);
      if ( v6 > 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_5;
      goto LABEL_8;
    }
  }
  v5 = off_49D8C00[v4];
  v6 = strlen(v5);
  if ( v6 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
LABEL_5:
    sub_4262D8((__int64)"basic_string::append");
LABEL_8:
  sub_2241490((unsigned __int64 *)a1, v5, v6);
  return a1;
}
