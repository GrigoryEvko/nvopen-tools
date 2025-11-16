// Function: sub_969EE0
// Address: 0x969ee0
//
void __fastcall sub_969EE0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // r12
  _QWORD *v4; // rdi
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // rdx
  _QWORD *v10; // r15
  _QWORD *v11; // rdx
  _QWORD *v12; // r14
  _QWORD *v14; // [rsp+28h] [rbp-48h]

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 )
  {
    v2 = 24LL * *(_QWORD *)(v1 - 8);
    v14 = (_QWORD *)(v1 + v2);
    if ( v1 == v1 + v2 )
    {
LABEL_35:
      j_j_j___libc_free_0_0(v14 - 1);
      return;
    }
    v3 = sub_C33340();
LABEL_6:
    while ( 2 )
    {
      v4 = v14 - 3;
      v14 = v4;
      if ( *v4 == v3 )
      {
        v5 = (_QWORD *)v4[1];
        if ( v5 )
        {
          v6 = &v5[3 * *(v5 - 1)];
          if ( v5 == v6 )
            goto LABEL_34;
          while ( 1 )
          {
            while ( 1 )
            {
              v6 -= 3;
              if ( v3 == *v6 )
                break;
              sub_C338F0(v6);
LABEL_11:
              if ( (_QWORD *)v4[1] == v6 )
                goto LABEL_34;
            }
            v7 = (_QWORD *)v6[1];
            if ( !v7 )
              goto LABEL_11;
            v8 = &v7[3 * *(v7 - 1)];
            if ( v7 != v8 )
            {
              do
              {
                while ( 1 )
                {
                  v8 -= 3;
                  if ( v3 == *v8 )
                    break;
                  sub_C338F0(v8);
LABEL_17:
                  if ( (_QWORD *)v6[1] == v8 )
                    goto LABEL_33;
                }
                v9 = (_QWORD *)v8[1];
                if ( !v9 )
                  goto LABEL_17;
                v10 = &v9[3 * *(v9 - 1)];
                if ( v9 != v10 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v10 -= 3;
                      if ( v3 == *v10 )
                        break;
                      sub_C338F0(v10);
LABEL_23:
                      if ( (_QWORD *)v8[1] == v10 )
                        goto LABEL_32;
                    }
                    v11 = (_QWORD *)v10[1];
                    if ( !v11 )
                      goto LABEL_23;
                    v12 = &v11[3 * *(v11 - 1)];
                    if ( v11 != v12 )
                    {
                      do
                      {
                        while ( 1 )
                        {
                          v12 -= 3;
                          if ( v3 == *v12 )
                            break;
                          sub_C338F0(v12);
                          if ( (_QWORD *)v10[1] == v12 )
                            goto LABEL_31;
                        }
                        sub_969EE0(v12);
                      }
                      while ( (_QWORD *)v10[1] != v12 );
                    }
LABEL_31:
                    j_j_j___libc_free_0_0(v12 - 1);
                  }
                  while ( (_QWORD *)v8[1] != v10 );
                }
LABEL_32:
                j_j_j___libc_free_0_0(v10 - 1);
              }
              while ( (_QWORD *)v6[1] != v8 );
            }
LABEL_33:
            j_j_j___libc_free_0_0(v8 - 1);
            if ( (_QWORD *)v4[1] == v6 )
            {
LABEL_34:
              j_j_j___libc_free_0_0(v6 - 1);
              if ( *(_QWORD **)(a1 + 8) == v4 )
                goto LABEL_35;
              goto LABEL_6;
            }
          }
        }
      }
      else
      {
        sub_C338F0(v4);
      }
      if ( *(_QWORD **)(a1 + 8) == v4 )
        goto LABEL_35;
      continue;
    }
  }
}
