// Function: sub_2BDFA30
// Address: 0x2bdfa30
//
char *__fastcall sub_2BDFA30(_QWORD *a1, unsigned __int8 a2)
{
  unsigned __int64 *v3; // rdi
  _BYTE *v4; // rax
  char *result; // rax
  char *v6; // rdx
  __int64 v7; // rax
  bool v8; // zf
  char v9; // r10
  __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rdx
  char v14; // [rsp+7h] [rbp-39h]

  v3 = a1 + 25;
  v4 = (_BYTE *)*v3;
  v3[1] = 0;
  *v4 = 0;
  result = (char *)*(v3 - 3);
  v6 = (char *)*(v3 - 2);
  if ( result == v6 )
    goto LABEL_13;
  while ( 1 )
  {
    v8 = *result == (char)a2;
    a1[22] = result + 1;
    if ( v8 )
      break;
    v9 = *result;
    v10 = a1[26];
    v11 = (_QWORD *)a1[25];
    v12 = v10 + 1;
    if ( v11 == a1 + 27 )
      v13 = 15;
    else
      v13 = a1[27];
    if ( v12 > v13 )
    {
      v14 = v9;
      sub_2240BB0(v3, a1[26], 0, 0, 1u);
      v11 = (_QWORD *)a1[25];
      v9 = v14;
    }
    *((_BYTE *)v11 + v10) = v9;
    v7 = a1[25];
    a1[26] = v12;
    *(_BYTE *)(v7 + v10 + 1) = 0;
    result = (char *)a1[22];
    v6 = (char *)a1[23];
    if ( result == v6 )
      goto LABEL_13;
  }
  if ( *result != a2 || v6 == result + 1 || (a1[22] = result + 2, result[1] != 93) )
LABEL_13:
    abort();
  return result;
}
