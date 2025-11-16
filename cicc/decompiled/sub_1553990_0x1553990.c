// Function: sub_1553990
// Address: 0x1553990
//
_BYTE *__fastcall sub_1553990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  void *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned __int8 **v14; // r12
  _WORD *v15; // rdx
  unsigned __int8 *v16; // rsi
  _DWORD *v17; // rdx
  _BYTE *result; // rax
  char v20; // [rsp+Fh] [rbp-61h]
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  char v22; // [rsp+18h] [rbp-58h]
  const char *v23; // [rsp+20h] [rbp-50h]
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]
  __int64 v26; // [rsp+38h] [rbp-38h]

  v7 = a1;
  v9 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v9 <= 0xEu )
  {
    sub_16E7EE0(a1, "!GenericDINode(", 15);
  }
  else
  {
    qmemcpy(v9, "!GenericDINode(", 15);
    *(_QWORD *)(a1 + 24) += 15LL;
  }
  v21 = a1;
  v23 = ", ";
  v22 = 1;
  v24 = a3;
  v25 = a4;
  v26 = a5;
  sub_1549850(&v21, a2);
  v10 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( v10 )
  {
    v10 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
    v12 = v11;
  }
  else
  {
    v12 = 0;
  }
  sub_154AC80(&v21, "header", 6u, v10, v12, 1);
  if ( *(_DWORD *)(a2 + 8) != 1 )
  {
    if ( v22 )
      v22 = 0;
    else
      a1 = sub_1263B40(a1, v23);
    sub_1263B40(a1, "operands: {");
    v13 = *(unsigned int *)(a2 + 8);
    if ( a2 + 8 * (1 - v13) != a2 )
    {
      v20 = 1;
      v14 = (unsigned __int8 **)(a2 + 8 * (1 - v13));
      while ( 1 )
      {
        if ( v20 )
        {
          v16 = *v14;
          v20 = 0;
          if ( !*v14 )
            goto LABEL_17;
LABEL_13:
          sub_154F770(v7, v16, a3, a4, a5);
LABEL_14:
          if ( (unsigned __int8 **)a2 == ++v14 )
            break;
        }
        else
        {
          v15 = *(_WORD **)(v7 + 24);
          if ( *(_QWORD *)(v7 + 16) - (_QWORD)v15 <= 1u )
          {
            sub_16E7EE0(v7, ", ", 2);
          }
          else
          {
            *v15 = 8236;
            *(_QWORD *)(v7 + 24) += 2LL;
          }
          v16 = *v14;
          if ( *v14 )
            goto LABEL_13;
LABEL_17:
          v17 = *(_DWORD **)(v7 + 24);
          if ( *(_QWORD *)(v7 + 16) - (_QWORD)v17 <= 3u )
          {
            sub_16E7EE0(v7, "null", 4);
            goto LABEL_14;
          }
          ++v14;
          *v17 = 1819047278;
          *(_QWORD *)(v7 + 24) += 4LL;
          if ( (unsigned __int8 **)a2 == v14 )
            break;
        }
      }
    }
    sub_1263B40(v7, "}");
  }
  result = *(_BYTE **)(v7 + 24);
  if ( *(_BYTE **)(v7 + 16) == result )
    return (_BYTE *)sub_16E7EE0(v7, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v7 + 24);
  return result;
}
