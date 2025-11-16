// Function: sub_15E92E0
// Address: 0x15e92e0
//
unsigned __int8 *__fastcall sub_15E92E0(__int64 a1, __int64 a2, unsigned __int8 **a3, char a4)
{
  _QWORD *v6; // rdi
  unsigned __int8 *v7; // r13
  size_t v8; // r12
  unsigned __int8 *result; // rax
  __int64 v10; // rax
  size_t v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = (_QWORD *)(a1 + 24);
  *(v6 - 3) = a2;
  *(_QWORD *)(a1 + 8) = v6;
  v7 = *a3;
  v8 = (size_t)a3[1];
  result = &(*a3)[v8];
  if ( result && !v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11[0] = (size_t)a3[1];
  if ( v8 > 0xF )
  {
    v10 = sub_22409D0(a1 + 8, v11, 0);
    *(_QWORD *)(a1 + 8) = v10;
    v6 = (_QWORD *)v10;
    *(_QWORD *)(a1 + 24) = v11[0];
    goto LABEL_10;
  }
  if ( v8 != 1 )
  {
    if ( !v8 )
      goto LABEL_6;
LABEL_10:
    result = (unsigned __int8 *)memcpy(v6, v7, v8);
    v8 = v11[0];
    v6 = *(_QWORD **)(a1 + 8);
    goto LABEL_6;
  }
  result = (unsigned __int8 *)*v7;
  *(_BYTE *)(a1 + 24) = (_BYTE)result;
LABEL_6:
  *(_QWORD *)(a1 + 16) = v8;
  *((_BYTE *)v6 + v8) = 0;
  *(_BYTE *)(a1 + 40) = a4;
  return result;
}
