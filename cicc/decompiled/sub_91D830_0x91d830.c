// Function: sub_91D830
// Address: 0x91d830
//
__int64 __fastcall sub_91D830(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rsi
  _QWORD *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // r14
  _QWORD *v18; // [rsp+28h] [rbp-48h]

  v1 = sub_C33340();
  if ( *a1 != v1 )
    return sub_C338F0(a1);
  v3 = v1;
  result = (__int64)a1;
  v4 = a1[1];
  if ( v4 )
  {
    v5 = 24LL * *(_QWORD *)(v4 - 8);
    v18 = (_QWORD *)(v4 + v5);
    if ( v4 == v4 + v5 )
      return j_j_j___libc_free_0_0(v18 - 1);
LABEL_8:
    while ( 2 )
    {
      v18 -= 3;
      if ( v3 == *v18 )
      {
        v6 = v18[1];
        if ( v6 )
        {
          v7 = 24LL * *(_QWORD *)(v6 - 8);
          v8 = (_QWORD *)(v6 + v7);
          if ( v6 == v6 + v7 )
            goto LABEL_33;
          while ( 1 )
          {
            while ( 1 )
            {
              v8 -= 3;
              if ( v3 == *v8 )
                break;
              sub_C338F0(v8);
LABEL_13:
              if ( (_QWORD *)v18[1] == v8 )
                goto LABEL_33;
            }
            v9 = v8[1];
            if ( !v9 )
              goto LABEL_13;
            v10 = 24LL * *(_QWORD *)(v9 - 8);
            v11 = (_QWORD *)(v9 + v10);
            if ( v9 != v9 + v10 )
            {
              do
              {
                while ( 1 )
                {
                  v11 -= 3;
                  if ( v3 == *v11 )
                    break;
                  sub_C338F0(v11);
LABEL_19:
                  if ( (_QWORD *)v8[1] == v11 )
                    goto LABEL_32;
                }
                v12 = v11[1];
                if ( !v12 )
                  goto LABEL_19;
                v13 = 24LL * *(_QWORD *)(v12 - 8);
                v14 = (_QWORD *)(v12 + v13);
                if ( v12 != v12 + v13 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v14 -= 3;
                      if ( v3 == *v14 )
                        break;
                      sub_C338F0(v14);
LABEL_25:
                      if ( (_QWORD *)v11[1] == v14 )
                        goto LABEL_31;
                    }
                    v15 = v14[1];
                    if ( !v15 )
                      goto LABEL_25;
                    v16 = 24LL * *(_QWORD *)(v15 - 8);
                    v17 = v15 + v16;
                    if ( v15 != v15 + v16 )
                    {
                      do
                      {
                        v17 -= 24;
                        sub_91D830(v17);
                      }
                      while ( v14[1] != v17 );
                    }
                    j_j_j___libc_free_0_0(v17 - 8);
                  }
                  while ( (_QWORD *)v11[1] != v14 );
                }
LABEL_31:
                j_j_j___libc_free_0_0(v14 - 1);
              }
              while ( (_QWORD *)v8[1] != v11 );
            }
LABEL_32:
            j_j_j___libc_free_0_0(v11 - 1);
            if ( (_QWORD *)v18[1] == v8 )
            {
LABEL_33:
              j_j_j___libc_free_0_0(v8 - 1);
              if ( (_QWORD *)a1[1] == v18 )
                return j_j_j___libc_free_0_0(v18 - 1);
              goto LABEL_8;
            }
          }
        }
      }
      else
      {
        ((void (*)(void))sub_C338F0)();
      }
      if ( (_QWORD *)a1[1] == v18 )
        return j_j_j___libc_free_0_0(v18 - 1);
      continue;
    }
  }
  return result;
}
