// Function: sub_1F3CD70
// Address: 0x1f3cd70
//
__int64 __fastcall sub_1F3CD70(char *a1, __int64 a2, signed __int64 *a3, unsigned __int8 *a4)
{
  size_t v5; // rdx
  _BYTE *v7; // rax
  signed __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  char *v12; // rax
  unsigned __int8 v13; // al

  if ( !a2 )
    goto LABEL_9;
  v5 = 0x7FFFFFFFFFFFFFFFLL;
  if ( a2 >= 0 )
    v5 = a2;
  v7 = memchr(a1, 58, v5);
  if ( v7 )
  {
    v8 = v7 - a1;
    *a3 = v8;
    if ( v8 == -1 )
    {
      return 0;
    }
    else
    {
      v9 = v8 + 1;
      if ( v9 > a2 || (v10 = a2 - v9, a2 - v9 == -1) || (v12 = &a1[v9], v10 != 1) || (v13 = *v12 - 48, v13 > 9u) )
        sub_16BD130("Invalid refinement step for -recip.", 1u);
      *a4 = v13;
      return 1;
    }
  }
  else
  {
LABEL_9:
    *a3 = -1;
    return 0;
  }
}
