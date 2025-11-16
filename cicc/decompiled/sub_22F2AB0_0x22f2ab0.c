// Function: sub_22F2AB0
// Address: 0x22f2ab0
//
unsigned __int8 *__fastcall sub_22F2AB0(__int64 a1, __int64 a2, unsigned __int8 **a3, char a4, char a5)
{
  _QWORD *v8; // rdi
  unsigned __int8 *v9; // r13
  size_t v10; // r12
  unsigned __int8 *result; // rax
  __int64 v12; // rax
  size_t v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = (_QWORD *)(a1 + 24);
  *(v8 - 3) = a2;
  *(_QWORD *)(a1 + 8) = v8;
  v9 = *a3;
  v10 = (size_t)a3[1];
  result = &(*a3)[v10];
  if ( result && !v9 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v13[0] = (size_t)a3[1];
  if ( v10 > 0xF )
  {
    v12 = sub_22409D0(a1 + 8, v13, 0);
    *(_QWORD *)(a1 + 8) = v12;
    v8 = (_QWORD *)v12;
    *(_QWORD *)(a1 + 24) = v13[0];
    goto LABEL_10;
  }
  if ( v10 != 1 )
  {
    if ( !v10 )
      goto LABEL_6;
LABEL_10:
    result = (unsigned __int8 *)memcpy(v8, v9, v10);
    v10 = v13[0];
    v8 = *(_QWORD **)(a1 + 8);
    goto LABEL_6;
  }
  result = (unsigned __int8 *)*v9;
  *(_BYTE *)(a1 + 24) = (_BYTE)result;
LABEL_6:
  *(_QWORD *)(a1 + 16) = v10;
  *((_BYTE *)v8 + v10) = 0;
  *(_BYTE *)(a1 + 40) = a4;
  *(_BYTE *)(a1 + 41) = a5;
  return result;
}
