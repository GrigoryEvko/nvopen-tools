// Function: sub_D23A60
// Address: 0xd23a60
//
char *__fastcall sub_D23A60(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *v8; // rbx
  __int64 *v9; // r15
  __int64 *v10; // r12
  bool v11; // zf
  __int64 v12; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r12
  char *v19; // r15
  size_t v20; // rdx
  char *v21; // [rsp+8h] [rbp-48h]
  char *v22; // [rsp+10h] [rbp-40h]

  if ( a4 == 1 )
    return (char *)a1;
  if ( a4 <= a6 )
  {
    v8 = a1 + 1;
    v9 = a5 + 1;
    v10 = a1;
    *a5 = *a1;
    if ( a2 == a1 + 1 )
    {
      v20 = 8;
    }
    else
    {
      do
      {
        while ( 1 )
        {
          v11 = (unsigned __int8)sub_B19060(a3, *v8, a3, a4) == 0;
          v12 = *v8;
          if ( v11 )
            break;
          ++v8;
          *v10++ = v12;
          if ( a2 == v8 )
            goto LABEL_8;
        }
        ++v8;
        *v9++ = v12;
      }
      while ( a2 != v8 );
LABEL_8:
      v20 = (char *)v9 - (char *)a5;
    }
    if ( a5 != v9 )
      memmove(v10, a5, v20);
    return (char *)v10;
  }
  v14 = a4 / 2;
  v22 = (char *)&a1[a4 / 2];
  v15 = sub_D23A60(a1, &a1[v14], a3, v14, a5);
  v18 = a4 - v14;
  v19 = v22;
  v21 = (char *)v15;
  if ( v18 )
  {
    while ( (unsigned __int8)sub_B19060(a3, *(_QWORD *)v19, v16, v17) )
    {
      v19 += 8;
      if ( !--v18 )
        return sub_D23650(v21, v22, v19);
    }
    v19 = (char *)sub_D23A60(v19, a2, a3, v18, a5);
  }
  return sub_D23650(v21, v22, v19);
}
