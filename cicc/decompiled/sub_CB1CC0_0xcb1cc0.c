// Function: sub_CB1CC0
// Address: 0xcb1cc0
//
void *__fastcall sub_CB1CC0(__int64 a1, char *a2, size_t a3, int a4)
{
  __int64 v4; // r13
  char *v5; // r12
  char *v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rbx
  int v10; // esi
  size_t v11; // rdx
  size_t v12; // rdx
  char *v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+18h] [rbp-58h]
  const void *v16[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF

  v4 = a3;
  v5 = a2;
  if ( a4 )
  {
    v7 = a2;
    if ( a4 == 1 )
    {
      v14 = "'";
      sub_CB1B10(a1, "'", 1u);
    }
    else
    {
      v14 = "\"";
      sub_CB1B10(a1, "\"", 1u);
      if ( a4 == 2 )
      {
        sub_CA6FB0(v16, a2, v4, 0);
        sub_CB1B10(a1, v16[0], (size_t)v16[1]);
        if ( v16[0] != &v17 )
          j_j___libc_free_0(v16[0], v17 + 1);
        a2 = "\"";
LABEL_11:
        a3 = 1;
        return sub_CB1B10(a1, a2, a3);
      }
    }
    if ( (_DWORD)v4 )
    {
      v8 = (unsigned int)v4;
      v9 = 0;
      v10 = 0;
      do
      {
        while ( v5[v9] != 39 )
        {
          if ( v8 == ++v9 )
            goto LABEL_9;
        }
        v11 = (unsigned int)(v9 - v10);
        v15 = v8;
        ++v9;
        sub_CB1B10(a1, &v5[v10], v11);
        sub_CB1B10(a1, "''", 2u);
        v8 = v15;
        v10 = v9;
      }
      while ( v15 != v9 );
LABEL_9:
      v7 = &v5[v10];
      v12 = (unsigned int)(v4 - v10);
    }
    else
    {
      v12 = 0;
    }
    sub_CB1B10(a1, v7, v12);
    a2 = v14;
    goto LABEL_11;
  }
  return sub_CB1B10(a1, a2, a3);
}
