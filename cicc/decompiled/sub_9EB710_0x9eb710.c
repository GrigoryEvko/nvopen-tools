// Function: sub_9EB710
// Address: 0x9eb710
//
char *__fastcall sub_9EB710(char *a1, char *a2, size_t a3, const void **a4)
{
  char *v4; // r14
  char *v5; // r12
  char *v6; // rdi
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r15
  __int64 v9; // rbx
  char *v10; // r13
  char *i; // r15
  char v12; // dl
  unsigned __int64 v13; // r15
  _QWORD *v14; // rbx
  const void **v15; // r15
  const void **v16; // r13
  __int64 v17; // rax
  char *v18; // rdi
  char *v19; // rax
  char *v21; // [rsp+0h] [rbp-40h]
  size_t v22; // [rsp+8h] [rbp-38h]
  size_t v23; // [rsp+8h] [rbp-38h]

  v4 = (char *)a3;
  v21 = a2;
  if ( a1 != a2 )
  {
    v5 = a1;
    do
    {
      if ( v4 )
      {
        v6 = v4 + 24;
        *((_QWORD *)v4 + 1) = 0;
        *(_QWORD *)v4 = v4 + 24;
        *((_QWORD *)v4 + 2) = 40;
        v7 = *((_QWORD *)v5 + 1);
        if ( v7 && v4 != v5 )
        {
          a3 = *((_QWORD *)v5 + 1);
          if ( v7 <= 0x28
            || (a2 = v4 + 24, sub_C8D290(v4, v4 + 24, a3, 1), a3 = *((_QWORD *)v5 + 1), v6 = *(char **)v4, a3) )
          {
            a2 = *(char **)v5;
            memcpy(v6, *(const void **)v5, a3);
          }
          *((_QWORD *)v4 + 1) = v7;
        }
        v8 = *((_QWORD *)v5 + 9) - *((_QWORD *)v5 + 8);
        *((_QWORD *)v4 + 8) = 0;
        *((_QWORD *)v4 + 9) = 0;
        *((_QWORD *)v4 + 10) = 0;
        if ( v8 )
        {
          if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_39;
          v6 = (char *)v8;
          v9 = sub_22077B0(v8);
        }
        else
        {
          v9 = 0;
        }
        *((_QWORD *)v4 + 8) = v9;
        *((_QWORD *)v4 + 9) = v9;
        *((_QWORD *)v4 + 10) = v9 + v8;
        v10 = (char *)*((_QWORD *)v5 + 9);
        for ( i = (char *)*((_QWORD *)v5 + 8); v10 != i; i += 72 )
        {
          while ( 1 )
          {
            if ( v9 )
            {
              v12 = *i;
              *(_DWORD *)(v9 + 16) = 0;
              *(_DWORD *)(v9 + 20) = 12;
              *(_BYTE *)v9 = v12;
              a3 = v9 + 24;
              *(_QWORD *)(v9 + 8) = v9 + 24;
              if ( *((_DWORD *)i + 4) )
                break;
            }
            i += 72;
            v9 += 72;
            if ( v10 == i )
              goto LABEL_14;
          }
          a2 = i + 8;
          v6 = (char *)(v9 + 8);
          v9 += 72;
          sub_9C2E20((__int64)v6, (__int64)a2);
        }
LABEL_14:
        *((_QWORD *)v4 + 9) = v9;
        v13 = *((_QWORD *)v5 + 12) - *((_QWORD *)v5 + 11);
        *((_QWORD *)v4 + 11) = 0;
        *((_QWORD *)v4 + 12) = 0;
        *((_QWORD *)v4 + 13) = 0;
        if ( v13 )
        {
          if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_39:
            sub_4261EA(v6, a2, a3, a4);
          v6 = (char *)v13;
          v14 = (_QWORD *)sub_22077B0(v13);
        }
        else
        {
          v13 = 0;
          v14 = 0;
        }
        *((_QWORD *)v4 + 11) = v14;
        *((_QWORD *)v4 + 12) = v14;
        *((_QWORD *)v4 + 13) = (char *)v14 + v13;
        v15 = (const void **)*((_QWORD *)v5 + 12);
        a4 = (const void **)*((_QWORD *)v5 + 11);
        if ( v15 != a4 )
        {
          v16 = (const void **)*((_QWORD *)v5 + 11);
          do
          {
            if ( v14 )
            {
              a3 = (_BYTE *)v16[1] - (_BYTE *)*v16;
              *v14 = 0;
              v14[1] = 0;
              v14[2] = 0;
              if ( a3 )
              {
                if ( a3 > 0x7FFFFFFFFFFFFFF0LL )
                  goto LABEL_39;
                v22 = a3;
                v17 = sub_22077B0(a3);
                a3 = v22;
                v18 = (char *)v17;
              }
              else
              {
                v18 = 0;
              }
              *v14 = v18;
              v14[2] = &v18[a3];
              v14[1] = v18;
              a2 = (char *)*v16;
              a3 = (_BYTE *)v16[1] - (_BYTE *)*v16;
              if ( v16[1] != *v16 )
              {
                v23 = (_BYTE *)v16[1] - (_BYTE *)*v16;
                v19 = (char *)memmove(v18, a2, a3);
                a3 = v23;
                v18 = v19;
              }
              v6 = &v18[a3];
              v14[1] = v6;
            }
            v16 += 3;
            v14 += 3;
          }
          while ( v15 != v16 );
        }
        *((_QWORD *)v4 + 12) = v14;
      }
      v5 += 112;
      v4 += 112;
    }
    while ( v21 != v5 );
  }
  return v4;
}
