// Function: sub_1407F20
// Address: 0x1407f20
//
__int64 __fastcall sub_1407F20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // r15
  __int64 i; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 *v16; // rcx
  __int64 v17; // r10
  __int64 *v18; // rbx
  _DWORD *v19; // rdx
  __int64 v20; // r12
  unsigned __int64 v21; // r13
  __int64 v22; // r14
  const char *v23; // r14
  size_t v24; // rax
  char *v25; // rdi
  size_t v26; // rdx
  char *v27; // rax
  _WORD *v28; // rdx
  __int64 v29; // rax
  int v30; // ecx
  int v31; // r8d
  __int64 v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  size_t v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  __int64 *v38; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 160);
  v3 = *(_QWORD *)(v2 + 80);
  v37 = v2 + 72;
  if ( v2 + 72 == v3 )
  {
    v4 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( v4 != v3 + 16 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 + 72 == v3 )
        break;
      if ( !v3 )
        BUG();
    }
  }
  result = a2;
  v6 = v3;
  i = v4;
  v8 = a2;
LABEL_8:
  if ( v6 != v37 )
  {
    v9 = v8;
    v10 = v6;
    v11 = v9;
    while ( 2 )
    {
      v12 = i - 24;
      if ( !i )
        v12 = 0;
      v13 = *(unsigned int *)(a1 + 192);
      v35 = v12;
      if ( (_DWORD)v13 )
      {
        v14 = *(_QWORD *)(a1 + 176);
        v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v16 = (__int64 *)(v14 + 168LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
        {
LABEL_14:
          if ( v16 != (__int64 *)(v14 + 168 * v13) )
          {
            v18 = (__int64 *)v16[11];
            v38 = &v18[2 * *((unsigned int *)v16 + 24)];
            if ( v38 != v18 )
            {
              v33 = i;
              v32 = v10;
              while ( 1 )
              {
                v19 = *(_DWORD **)(v11 + 24);
                v20 = v18[1];
                v21 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
                v22 = (*v18 >> 1) & 3;
                if ( *(_QWORD *)(v11 + 16) - (_QWORD)v19 <= 3u )
                {
                  sub_16E7EE0(v11, "    ", 4);
                }
                else
                {
                  *v19 = 538976288;
                  *(_QWORD *)(v11 + 24) += 4LL;
                }
                v23 = off_4984800[v22];
                if ( v23 )
                {
                  v24 = strlen(v23);
                  v25 = *(char **)(v11 + 24);
                  v26 = v24;
                  v27 = *(char **)(v11 + 16);
                  if ( v26 <= v27 - v25 )
                  {
                    if ( v26 )
                    {
                      v36 = v26;
                      memcpy(v25, v23, v26);
                      v27 = *(char **)(v11 + 16);
                      v25 = (char *)(v36 + *(_QWORD *)(v11 + 24));
                      *(_QWORD *)(v11 + 24) = v25;
                    }
                    goto LABEL_25;
                  }
                  sub_16E7EE0(v11, v23);
                }
                v27 = *(char **)(v11 + 16);
                v25 = *(char **)(v11 + 24);
LABEL_25:
                if ( v20 )
                {
                  if ( (unsigned __int64)(v27 - v25) <= 9 )
                  {
                    sub_16E7EE0(v11, " in block ", 10);
                  }
                  else
                  {
                    qmemcpy(v25, " in block ", 10);
                    *(_QWORD *)(v11 + 24) += 10LL;
                  }
                  sub_15537D0(v20, v11, 0);
                  v27 = *(char **)(v11 + 16);
                  v25 = *(char **)(v11 + 24);
                }
                if ( v21 )
                {
                  if ( (unsigned __int64)(v27 - v25) <= 6 )
                  {
                    sub_16E7EE0(v11, " from: ", 7);
                  }
                  else
                  {
                    *(_DWORD *)v25 = 1869768224;
                    *((_WORD *)v25 + 2) = 14957;
                    v25[6] = 32;
                    *(_QWORD *)(v11 + 24) += 7LL;
                  }
                  sub_155C2B0(v21, v11, 0);
                  v25 = *(char **)(v11 + 24);
                  if ( v25 == *(char **)(v11 + 16) )
                    goto LABEL_33;
LABEL_18:
                  *v25 = 10;
                  v18 += 2;
                  ++*(_QWORD *)(v11 + 24);
                  if ( v38 == v18 )
                    goto LABEL_34;
                }
                else
                {
                  if ( v25 != v27 )
                    goto LABEL_18;
LABEL_33:
                  v18 += 2;
                  sub_16E7EE0(v11, "\n", 1);
                  if ( v38 == v18 )
                  {
LABEL_34:
                    i = v33;
                    v10 = v32;
                    break;
                  }
                }
              }
            }
            sub_155C2B0(v35, v11, 0);
            v28 = *(_WORD **)(v11 + 24);
            if ( *(_QWORD *)(v11 + 16) - (_QWORD)v28 <= 1u )
            {
              sub_16E7EE0(v11, "\n\n", 2);
            }
            else
            {
              *v28 = 2570;
              *(_QWORD *)(v11 + 24) += 2LL;
            }
          }
        }
        else
        {
          v30 = 1;
          while ( v17 != -8 )
          {
            v31 = v30 + 1;
            v15 = (v13 - 1) & (v30 + v15);
            v16 = (__int64 *)(v14 + 168LL * v15);
            v17 = *v16;
            if ( v12 == *v16 )
              goto LABEL_14;
            v30 = v31;
          }
        }
      }
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v10 + 24) )
      {
        v29 = v10 - 24;
        if ( !v10 )
          v29 = 0;
        result = v29 + 40;
        if ( i != result )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( v37 == v10 )
        {
          result = v11;
          v6 = v10;
          v8 = result;
          goto LABEL_8;
        }
        if ( !v10 )
          BUG();
      }
      if ( v37 != v10 )
        continue;
      break;
    }
  }
  return result;
}
