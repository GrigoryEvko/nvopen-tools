// Function: sub_30289C0
// Address: 0x30289c0
//
_BYTE *__fastcall sub_30289C0(_QWORD *a1, char a2, __int64 a3)
{
  __int64 v4; // rax
  _BYTE *v5; // r15
  _BYTE *result; // rax
  size_t v8; // r14
  char v9; // al
  char *v10; // rdx
  char *v11; // rax
  char *v12; // rdx
  size_t v13; // rsi
  size_t v14; // rdi
  size_t v15; // rdi
  _BYTE *v16; // [rsp+8h] [rbp-38h]

  v4 = a1[1];
  if ( !v4 )
    return (_BYTE *)a3;
  v5 = (_BYTE *)*a1;
  if ( a2 != *(_BYTE *)*a1 )
    return (_BYTE *)a3;
  v8 = v4 - 1;
  *a1 = v5 + 1;
  a1[1] = v4 - 1;
  if ( v4 == 1 )
    return (_BYTE *)a3;
  v9 = v5[1];
  v10 = "[]";
  if ( v9 != 91 )
  {
    v10 = "<>";
    if ( v9 != 60 )
    {
      v10 = "()";
      if ( v9 != 40 )
        return (_BYTE *)a3;
    }
  }
  v16 = v5 + 1;
  v11 = (char *)memchr(v5 + 1, v10[1], v8);
  if ( !v11 )
    return (_BYTE *)a3;
  v12 = (char *)(v11 - v16);
  if ( v11 - v16 == -1 )
    return (_BYTE *)a3;
  v13 = (size_t)(v12 + 1);
  result = v5 + 2;
  if ( !v12 || (v14 = 0, v13 <= v8) )
  {
    v15 = v8;
    v8 = (size_t)(v12 + 1);
    v14 = v15 - v13;
  }
  a1[1] = v14;
  *a1 = &v16[v8];
  return result;
}
