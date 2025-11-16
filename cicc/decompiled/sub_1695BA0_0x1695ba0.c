// Function: sub_1695BA0
// Address: 0x1695ba0
//
unsigned __int64 __fastcall sub_1695BA0(char **a1, unsigned __int64 a2)
{
  char *v2; // r15
  unsigned __int64 result; // rax
  char *v5; // r13
  __int64 v6; // rax
  char **v7; // rbx
  char *v8; // rdx
  char *v9; // r14
  char *v10; // rdi
  signed __int64 v11; // [rsp+10h] [rbp-40h]
  char **v12; // [rsp+18h] [rbp-38h]

  if ( a2 > 0x555555555555555LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *a1;
  result = 0xAAAAAAAAAAAAAAABLL * ((a1[2] - *a1) >> 3);
  if ( a2 > result )
  {
    v5 = a1[1];
    v12 = 0;
    v11 = v5 - v2;
    if ( a2 )
    {
      v6 = sub_22077B0(24 * a2);
      v5 = a1[1];
      v2 = *a1;
      v12 = (char **)v6;
    }
    if ( v5 != v2 )
    {
      v7 = v12;
      while ( 1 )
      {
        v9 = *(char **)v2;
        if ( !v7 )
          goto LABEL_11;
        *v7 = v9;
        v8 = (char *)*((_QWORD *)v2 + 1);
        v7[1] = v8;
        v7[2] = (char *)*((_QWORD *)v2 + 2);
        if ( v2 == v9 )
        {
          v7[1] = (char *)v7;
          *v7 = (char *)v7;
          v9 = *(char **)v2;
LABEL_11:
          if ( v9 == v2 )
            goto LABEL_9;
          do
          {
            v10 = v9;
            v9 = *(char **)v9;
            j_j___libc_free_0(v10, 32);
          }
          while ( v9 != v2 );
          v2 += 24;
          v7 += 3;
          if ( v5 == v2 )
          {
LABEL_14:
            v2 = *a1;
            break;
          }
        }
        else
        {
          *(_QWORD *)v8 = v7;
          *((_QWORD *)*v7 + 1) = v7;
          *((_QWORD *)v2 + 1) = v2;
          *(_QWORD *)v2 = v2;
          *((_QWORD *)v2 + 2) = 0;
LABEL_9:
          v2 += 24;
          v7 += 3;
          if ( v5 == v2 )
            goto LABEL_14;
        }
      }
    }
    if ( v2 )
      j_j___libc_free_0(v2, a1[2] - v2);
    *a1 = (char *)v12;
    result = (unsigned __int64)v12 + v11;
    a1[1] = (char *)v12 + v11;
    a1[2] = (char *)&v12[3 * a2];
  }
  return result;
}
