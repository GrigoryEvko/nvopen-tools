// Function: sub_C0A5E0
// Address: 0xc0a5e0
//
unsigned __int8 *__fastcall sub_C0A5E0(__int64 *a1, unsigned __int8 **a2)
{
  void *v3; // rdi
  unsigned __int8 *v4; // r13
  size_t v5; // r12
  unsigned __int8 *result; // rax
  __int64 v7; // rax
  size_t v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1 + 2;
  v4 = *a2;
  v5 = (size_t)a2[1];
  *a1 = (__int64)v3;
  result = &v4[v5];
  if ( &v4[v5] && !v4 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v8[0] = v5;
  if ( v5 > 0xF )
  {
    v7 = sub_22409D0(a1, v8, 0);
    *a1 = v7;
    v3 = (void *)v7;
    a1[2] = v8[0];
    goto LABEL_10;
  }
  if ( v5 != 1 )
  {
    if ( !v5 )
      goto LABEL_6;
LABEL_10:
    result = (unsigned __int8 *)memcpy(v3, v4, v5);
    v5 = v8[0];
    v3 = (void *)*a1;
    goto LABEL_6;
  }
  result = (unsigned __int8 *)*v4;
  *((_BYTE *)a1 + 16) = (_BYTE)result;
LABEL_6:
  a1[1] = v5;
  *((_BYTE *)v3 + v5) = 0;
  return result;
}
