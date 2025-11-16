// Function: sub_87F870
// Address: 0x87f870
//
_QWORD *__fastcall sub_87F870(__int64 *a1, _QWORD *a2, __int64 j, _QWORD *a4)
{
  _QWORD *v4; // r15
  __int64 *v5; // rax
  __int64 i; // rbx
  _QWORD *v7; // rax
  __int64 v8; // r13
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *result; // rax
  char v12; // cl
  char *v13; // rax
  _BYTE *v14; // [rsp+10h] [rbp-40h]
  char v15; // [rsp+1Ch] [rbp-34h]

  v4 = a4;
  v15 = j;
  if ( !a1 )
  {
    if ( !a2 )
      goto LABEL_11;
    i = 0;
    goto LABEL_5;
  }
  j = *a1;
  v5 = (__int64 *)a1[1];
  for ( i = *(_QWORD *)(*a1 + 16); v5; i += *(_QWORD *)(j + 16) + 1LL )
  {
    j = *v5;
    v5 = (__int64 *)v5[1];
  }
  if ( a2 )
  {
LABEL_5:
    a4 = (_QWORD *)*a2;
    v7 = (_QWORD *)a2[1];
    for ( j = *(_QWORD *)(*a2 + 16LL) + 1LL; v7; j += a4[2] + 1LL )
    {
      a4 = (_QWORD *)*v7;
      v7 = (_QWORD *)v7[1];
    }
    i += j;
  }
  if ( !i )
  {
LABEL_11:
    v8 = sub_877070(a1, a2, j, a4);
    v13 = (char *)sub_7279A0(10);
    strcpy(v13, "<unnamed>");
    *(_QWORD *)(v8 + 8) = v13;
    *(_BYTE *)(v8 + 73) |= 1u;
    *(_QWORD *)(v8 + 16) = 9;
    goto LABEL_10;
  }
  v8 = sub_877070(a1, a2, j, a4);
  v9 = (_BYTE *)sub_7279A0(i + 1);
  *(_QWORD *)(v8 + 8) = v9;
  v14 = v9;
  v10 = sub_877270(v9, a1, 0, i + 1);
  sub_877270(&v14[v10], a2, 1, i + 1 - v10);
  v14[i] = 0;
  *(_QWORD *)(v8 + 16) = i;
LABEL_10:
  result = sub_87EBB0(0x19u, v8, v4);
  v12 = *((_BYTE *)result + 104);
  result[11] = a1;
  result[12] = a2;
  *((_BYTE *)result + 104) = v12 & 0xFE | v15 & 1;
  return result;
}
