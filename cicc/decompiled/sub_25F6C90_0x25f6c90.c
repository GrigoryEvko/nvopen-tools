// Function: sub_25F6C90
// Address: 0x25f6c90
//
void __fastcall sub_25F6C90(unsigned int *src, unsigned int *a2, _DWORD *a3)
{
  __int64 v3; // rbx
  char *v4; // r14
  unsigned int *v5; // r12
  unsigned int *v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r15
  _DWORD *v10; // r8
  char *v11; // r14
  __int64 v12; // rbx
  int *v13; // rdi
  __int64 v14; // rax
  char *v15; // r8
  char *v16; // r14
  __int64 v17; // r12
  char *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+0h] [rbp-60h]
  char *v23; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = (char *)a2 - (char *)src;
  v4 = (char *)a3 + (char *)a2 - (char *)src;
  v26 = a2 - src;
  if ( (char *)a2 - (char *)src <= 24 )
  {
    sub_25F6BE0(src, a2);
  }
  else
  {
    v5 = src;
    do
    {
      v6 = v5;
      v5 += 7;
      sub_25F6BE0(v6, v5);
    }
    while ( (char *)a2 - (char *)v5 > 24 );
    sub_25F6BE0(v5, a2);
    if ( v3 > 28 )
    {
      v23 = v4;
      v7 = 7;
      while ( 1 )
      {
        v8 = 2 * v7;
        if ( v26 < 2 * v7 )
        {
          v10 = a3;
          v14 = v26;
          v11 = (char *)src;
        }
        else
        {
          v9 = 8 * v7;
          v10 = a3;
          v11 = (char *)src;
          v21 = v7;
          v12 = -4 * v7;
          do
          {
            v13 = (int *)v11;
            v11 += v9;
            v10 = (_DWORD *)sub_25F5CE0(v13, (int *)&v11[v12], &v11[v12], v11, v10);
            v14 = ((char *)a2 - v11) >> 2;
          }
          while ( v8 <= v14 );
          v7 = v21;
        }
        if ( v7 <= v14 )
          v14 = v7;
        v7 *= 4;
        sub_25F5CE0((int *)v11, (int *)&v11[4 * v14], &v11[4 * v14], (char *)a2, v10);
        v15 = (char *)src;
        if ( v26 < v7 )
          break;
        v22 = v8;
        v16 = (char *)a3;
        v17 = 4 * v8 - 4 * v7;
        do
        {
          v18 = v16;
          v16 += 4 * v7;
          v15 = sub_25F6270(v18, &v16[v17], &v16[v17], v16, v15);
          v19 = (v23 - v16) >> 2;
        }
        while ( v7 <= v19 );
        if ( v19 > v22 )
          v19 = v22;
        sub_25F6270(v16, &v16[4 * v19], &v16[4 * v19], v23, v15);
        if ( v26 <= v7 )
          return;
      }
      v20 = v8;
      if ( v26 <= v8 )
        v20 = v26;
      sub_25F6270(a3, &a3[v20], (char *)&a3[v20], v23, src);
    }
  }
}
