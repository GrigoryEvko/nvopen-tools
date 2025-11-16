// Function: sub_A52D40
// Address: 0xa52d40
//
__int64 __fastcall sub_A52D40(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char *v4; // rsi
  _QWORD *v5; // rax
  char v6; // r13
  _WORD *v7; // rdx
  __int64 v8; // rcx
  char *v9; // r8
  size_t v10; // rbx
  void *v11; // rdi
  __int64 v12; // r15
  _WORD *v13; // rdx
  __int64 v14; // rdi
  _WORD *v15; // rdx
  __int64 v16; // r15
  char *v17; // rax
  size_t v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  void *v21; // rdi
  __int64 v22; // rbx
  int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rbx
  __int64 v27; // rdi
  _WORD *v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // r13
  __int64 *v31; // rbx
  __int64 v32; // rsi
  _WORD *v33; // rdx
  size_t v34; // [rsp+0h] [rbp-50h]
  char *v35; // [rsp+0h] [rbp-50h]
  _QWORD *v36; // [rsp+8h] [rbp-48h]
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1;
  v4 = "!DIExpression(";
  sub_904010(a1, "!DIExpression(");
  if ( (unsigned __int8)sub_AF4230(a2) )
  {
    v5 = *(_QWORD **)(a2 + 16);
    v6 = 1;
    v36 = *(_QWORD **)(a2 + 24);
    v37[0] = v5;
    if ( v5 != v36 )
    {
      while ( 1 )
      {
        v9 = (char *)sub_E06E20(*v5);
        v10 = (size_t)v7;
        if ( v6 )
          break;
        v7 = *(_WORD **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v7 <= 1u )
        {
          v4 = ", ";
          v35 = v9;
          v29 = sub_CB6200(v2, ", ", 2);
          v9 = v35;
          v11 = *(void **)(v29 + 32);
          v12 = v29;
LABEL_5:
          if ( *(_QWORD *)(v12 + 24) - (_QWORD)v11 < v10 )
            goto LABEL_20;
          goto LABEL_6;
        }
        v12 = v2;
        *v7 = 8236;
        v11 = (void *)(*(_QWORD *)(v2 + 32) + 2LL);
        *(_QWORD *)(v2 + 32) = v11;
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v11 < v10 )
        {
LABEL_20:
          v4 = v9;
          sub_CB6200(v12, v9, v10);
          if ( *(_QWORD *)v37[0] != 4097 )
            goto LABEL_21;
          goto LABEL_9;
        }
LABEL_6:
        if ( v10 )
        {
          v4 = v9;
          memcpy(v11, v9, v10);
          *(_QWORD *)(v12 + 32) += v10;
        }
        if ( *(_QWORD *)v37[0] != 4097 )
        {
LABEL_21:
          v24 = sub_AF4160(v37, v4, v7, v8, v9);
          if ( v24 != 1 )
          {
            v25 = 8;
            v26 = 8LL * (unsigned int)(v24 - 2) + 16;
            do
            {
              v28 = *(_WORD **)(v2 + 32);
              if ( *(_QWORD *)(v2 + 24) - (_QWORD)v28 > 1u )
              {
                v27 = v2;
                *v28 = 8236;
                *(_QWORD *)(v2 + 32) += 2LL;
              }
              else
              {
                v27 = sub_CB6200(v2, ", ", 2);
              }
              v4 = *(char **)(v37[0] + v25);
              v25 += 8;
              sub_CB59D0(v27, v4);
            }
            while ( v26 != v25 );
          }
          goto LABEL_16;
        }
LABEL_9:
        v13 = *(_WORD **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v13 <= 1u )
        {
          v14 = sub_CB6200(v2, ", ", 2);
        }
        else
        {
          *v13 = 8236;
          v14 = v2;
          *(_QWORD *)(v2 + 32) += 2LL;
        }
        sub_CB59D0(v14, *(_QWORD *)(v37[0] + 8LL));
        v15 = *(_WORD **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v15 <= 1u )
        {
          v16 = sub_CB6200(v2, ", ", 2);
        }
        else
        {
          v16 = v2;
          *v15 = 8236;
          *(_QWORD *)(v2 + 32) += 2LL;
        }
        v17 = (char *)sub_E09D50(*(_QWORD *)(v37[0] + 16LL));
        v21 = *(void **)(v16 + 32);
        v4 = v17;
        if ( *(_QWORD *)(v16 + 24) - (_QWORD)v21 < v18 )
        {
          sub_CB6200(v16, v17, v18);
        }
        else if ( v18 )
        {
          v34 = v18;
          memcpy(v21, v17, v18);
          v18 = v34;
          *(_QWORD *)(v16 + 32) += v34;
        }
LABEL_16:
        v22 = v37[0];
        v5 = (_QWORD *)(v22 + 8LL * (unsigned int)sub_AF4160(v37, v4, v18, v19, v20));
        v37[0] = v5;
        if ( v36 == v5 )
          return sub_904010(v2, ")");
      }
      v11 = *(void **)(v2 + 32);
      v12 = v2;
      v6 = 0;
      goto LABEL_5;
    }
  }
  else
  {
    v30 = *(__int64 **)(a2 + 24);
    v31 = *(__int64 **)(a2 + 16);
    if ( v31 != v30 )
    {
      while ( 1 )
      {
        v32 = *v31++;
        sub_CB59D0(a1, v32);
        if ( v30 == v31 )
          break;
        v33 = *(_WORD **)(v2 + 32);
        if ( *(_QWORD *)(v2 + 24) - (_QWORD)v33 <= 1u )
        {
          a1 = sub_CB6200(v2, ", ", 2);
        }
        else
        {
          a1 = v2;
          *v33 = 8236;
          *(_QWORD *)(v2 + 32) += 2LL;
        }
      }
    }
  }
  return sub_904010(v2, ")");
}
