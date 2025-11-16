// Function: sub_A78010
// Address: 0xa78010
//
unsigned __int64 __fastcall sub_A78010(_QWORD *a1, int *a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // r14
  int v5; // eax
  unsigned __int64 v6; // rdx
  int v7; // r9d
  unsigned __int64 *v8; // r8
  __int64 v9; // r15
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // rdx
  int *i; // rcx
  int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned __int64 *v15; // rsi
  unsigned __int64 v16; // [rsp-70h] [rbp-70h]
  int v17; // [rsp-70h] [rbp-70h]
  unsigned __int64 *v18; // [rsp-68h] [rbp-68h] BYREF
  __int64 v19; // [rsp-60h] [rbp-60h]
  __int64 v20; // [rsp-58h] [rbp-58h] BYREF
  __int64 v21; // [rsp-50h] [rbp-50h] BYREF

  if ( !a3 )
    return 0;
  v4 = 4 * a3;
  v5 = a2[4 * a3 - 4];
  if ( v5 == -1 )
  {
    if ( a3 == 1 )
    {
      HIDWORD(v19) = 4;
      v7 = 1;
      v11 = (unsigned __int64 *)&v21;
      v18 = (unsigned __int64 *)&v20;
      v10 = (unsigned __int64 *)&v20;
      do
      {
LABEL_8:
        if ( v10 )
          *v10 = 0;
        ++v10;
      }
      while ( v10 != v11 );
      v8 = v18;
LABEL_12:
      LODWORD(v19) = v7;
      goto LABEL_13;
    }
    v5 = a2[v4 - 8];
  }
  v6 = (unsigned int)(v5 + 2);
  v18 = (unsigned __int64 *)&v20;
  v7 = v5 + 2;
  v8 = (unsigned __int64 *)&v20;
  v19 = 0x400000000LL;
  if ( v5 != -2 )
  {
    v9 = v6;
    v10 = (unsigned __int64 *)&v20;
    if ( v6 > 4 )
    {
      v17 = v6;
      sub_C8D5F0(&v18, &v20, v6, 8);
      v8 = v18;
      v7 = v17;
      v10 = &v18[(unsigned int)v19];
    }
    v11 = &v8[v9];
    if ( &v8[v9] == v10 )
      goto LABEL_12;
    goto LABEL_8;
  }
LABEL_13:
  for ( i = &a2[v4]; i != a2; v8 = v18 )
  {
    v13 = *a2;
    v14 = *((_QWORD *)a2 + 1);
    a2 += 4;
    v8[v13 + 1] = v14;
  }
  v15 = v8;
  result = sub_A77EC0(a1, v8, (unsigned int)v19);
  if ( v18 != (unsigned __int64 *)&v20 )
  {
    v16 = result;
    _libc_free(v18, v15);
    return v16;
  }
  return result;
}
