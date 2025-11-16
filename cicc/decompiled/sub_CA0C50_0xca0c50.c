// Function: sub_CA0C50
// Address: 0xca0c50
//
__int64 __fastcall sub_CA0C50(__int64 a1, _BYTE *a2, size_t a3)
{
  size_t v3; // rax
  _BYTE *v6; // rdi
  __int64 result; // rax
  bool v8; // zf
  __int64 v9; // rax
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a3;
  v6 = (_BYTE *)(a1 + 16);
  *(_QWORD *)a1 = v6;
  if ( &a2[a3] )
  {
    if ( !a2 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
  v10[0] = a3;
  if ( a3 > 0xF )
  {
    v9 = sub_22409D0(a1, v10, 0);
    *(_QWORD *)a1 = v9;
    v6 = (_BYTE *)v9;
    *(_QWORD *)(a1 + 16) = v10[0];
    goto LABEL_11;
  }
  if ( a3 != 1 )
  {
    if ( !a3 )
    {
LABEL_8:
      *(_QWORD *)(a1 + 8) = v3;
      v6[v3] = 0;
      *(_BYTE *)(a1 + 32) = 0;
      return sub_C8C750(a2, a3);
    }
LABEL_11:
    memcpy(v6, a2, a3);
    v3 = v10[0];
    v6 = *(_BYTE **)a1;
    goto LABEL_8;
  }
  result = (unsigned __int8)*a2;
  *(_BYTE *)(a1 + 17) = 0;
  v8 = *a2 == 45;
  *(_QWORD *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 16) = result;
  *(_BYTE *)(a1 + 32) = 0;
  if ( !v8 )
    return sub_C8C750(a2, a3);
  return result;
}
