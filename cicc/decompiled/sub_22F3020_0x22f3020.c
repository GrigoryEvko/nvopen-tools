// Function: sub_22F3020
// Address: 0x22f3020
//
unsigned __int8 *__fastcall sub_22F3020(_QWORD *a1, __int64 a2, unsigned __int8 **a3)
{
  _QWORD *v4; // rdi
  unsigned __int8 *v5; // r13
  size_t v6; // r12
  unsigned __int8 *result; // rax
  __int64 v8; // rax
  size_t v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a1 + 3;
  *(v4 - 3) = a2;
  a1[1] = v4;
  v5 = *a3;
  v6 = (size_t)a3[1];
  result = &(*a3)[v6];
  if ( result && !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v9[0] = (size_t)a3[1];
  if ( v6 > 0xF )
  {
    v8 = sub_22409D0((__int64)(a1 + 1), v9, 0);
    a1[1] = v8;
    v4 = (_QWORD *)v8;
    a1[3] = v9[0];
    goto LABEL_10;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
      goto LABEL_6;
LABEL_10:
    result = (unsigned __int8 *)memcpy(v4, v5, v6);
    v6 = v9[0];
    v4 = (_QWORD *)a1[1];
    goto LABEL_6;
  }
  result = (unsigned __int8 *)*v5;
  *((_BYTE *)a1 + 24) = (_BYTE)result;
LABEL_6:
  a1[2] = v6;
  *((_BYTE *)v4 + v6) = 0;
  return result;
}
