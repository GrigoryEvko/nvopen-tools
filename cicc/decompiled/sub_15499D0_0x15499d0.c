// Function: sub_15499D0
// Address: 0x15499d0
//
_BYTE *__fastcall sub_15499D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  void *v3; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // rcx
  char v6; // r14
  const char *v7; // rbx
  size_t v8; // rdx
  size_t v9; // r12
  void *v10; // rdi
  __int64 v11; // r9
  int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // rsi
  _WORD *v17; // rdx
  __int64 v18; // rbx
  _BYTE *result; // rax
  _WORD *v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // r12
  __int64 *v23; // rbx
  __int64 v24; // rsi
  _WORD *v25; // rdx
  __int64 v26; // [rsp+0h] [rbp-50h]
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  _QWORD v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v3 <= 0xDu )
  {
    sub_16E7EE0(a1, "!DIExpression(", 14);
  }
  else
  {
    qmemcpy(v3, "!DIExpression(", 14);
    *(_QWORD *)(a1 + 24) += 14LL;
  }
  if ( (unsigned __int8)sub_15B1200(a2) )
  {
    v4 = *(_QWORD **)(a2 + 24);
    v5 = *(_QWORD **)(a2 + 32);
    v6 = 1;
    v28[0] = v4;
    v27 = v5;
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v7 = sub_14E3970(*v4);
        v9 = v8;
        if ( v6 )
          break;
        v20 = *(_WORD **)(v2 + 24);
        if ( *(_QWORD *)(v2 + 16) - (_QWORD)v20 <= 1u )
        {
          v21 = sub_16E7EE0(v2, ", ", 2);
          v10 = *(void **)(v21 + 24);
          v11 = v21;
LABEL_7:
          if ( *(_QWORD *)(v11 + 16) - (_QWORD)v10 >= v9 )
            goto LABEL_8;
          goto LABEL_21;
        }
        v11 = v2;
        *v20 = 8236;
        v10 = (void *)(*(_QWORD *)(v2 + 24) + 2LL);
        *(_QWORD *)(v2 + 24) = v10;
        if ( *(_QWORD *)(v2 + 16) - (_QWORD)v10 >= v9 )
        {
LABEL_8:
          if ( v9 )
          {
            v26 = v11;
            memcpy(v10, v7, v9);
            *(_QWORD *)(v26 + 24) += v9;
          }
          goto LABEL_10;
        }
LABEL_21:
        sub_16E7EE0(v11, v7, v9);
LABEL_10:
        v12 = sub_15B11B0(v28);
        if ( v12 != 1 )
        {
          v13 = 8;
          v14 = 8LL * (unsigned int)(v12 - 2) + 16;
          do
          {
            v17 = *(_WORD **)(v2 + 24);
            if ( *(_QWORD *)(v2 + 16) - (_QWORD)v17 > 1u )
            {
              v15 = v2;
              *v17 = 8236;
              *(_QWORD *)(v2 + 24) += 2LL;
            }
            else
            {
              v15 = sub_16E7EE0(v2, ", ", 2);
            }
            v16 = *(_QWORD *)(v28[0] + v13);
            v13 += 8;
            sub_16E7A90(v15, v16);
          }
          while ( v14 != v13 );
        }
        v18 = v28[0];
        v4 = (_QWORD *)(v18 + 8LL * (unsigned int)sub_15B11B0(v28));
        v28[0] = v4;
        if ( v27 == v4 )
          goto LABEL_17;
      }
      v10 = *(void **)(v2 + 24);
      v11 = v2;
      v6 = 0;
      goto LABEL_7;
    }
  }
  else
  {
    v22 = *(__int64 **)(a2 + 32);
    v23 = *(__int64 **)(a2 + 24);
    if ( v23 != v22 )
    {
      while ( 1 )
      {
        v24 = *v23++;
        sub_16E7A90(a1, v24);
        if ( v22 == v23 )
          break;
        v25 = *(_WORD **)(v2 + 24);
        if ( *(_QWORD *)(v2 + 16) - (_QWORD)v25 <= 1u )
        {
          a1 = sub_16E7EE0(v2, ", ", 2);
        }
        else
        {
          a1 = v2;
          *v25 = 8236;
          *(_QWORD *)(v2 + 24) += 2LL;
        }
      }
    }
  }
LABEL_17:
  result = *(_BYTE **)(v2 + 24);
  if ( *(_BYTE **)(v2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(v2, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v2 + 24);
  return result;
}
