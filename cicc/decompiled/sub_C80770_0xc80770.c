// Function: sub_C80770
// Address: 0xc80770
//
__int64 __fastcall sub_C80770(char *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rcx
  __int64 v5; // rbx
  char *v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8; // [rsp+8h] [rbp-18h]

  v7 = a1;
  v8 = a2;
  if ( !a2 )
  {
LABEL_2:
    v4 = a2 - 1;
    if ( a3 > 1 )
    {
      v5 = sub_C93660(&v7, "\\/", 2, v4);
      if ( v5 != -1 )
        goto LABEL_4;
      v5 = v8 - 1;
      if ( v8 )
      {
        while ( v5 )
        {
          if ( v7[--v5] == 58 )
            goto LABEL_4;
        }
      }
    }
    else
    {
      v5 = sub_C93660(&v7, "/", 1, v4);
      if ( v5 != -1 )
      {
LABEL_4:
        if ( v5 != 1 || !sub_C80220(*v7, a3) )
          return v5 + 1;
      }
    }
    return 0;
  }
  if ( !sub_C80220(a1[a2 - 1], a3) )
  {
    a2 = v8;
    goto LABEL_2;
  }
  return v8 - 1;
}
