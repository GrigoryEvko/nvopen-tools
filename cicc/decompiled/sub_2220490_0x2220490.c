// Function: sub_2220490
// Address: 0x2220490
//
__int64 __fastcall sub_2220490(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, char a6, __int64 a7)
{
  _QWORD *v7; // rax
  _BYTE *v10; // r14
  size_t v11; // r12
  __int64 v12; // r12
  _QWORD *v14; // rdi
  size_t v18; // [rsp+28h] [rbp-68h] BYREF
  __int64 v19[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v20[2]; // [rsp+40h] [rbp-50h] BYREF
  void (__fastcall *v21)(unsigned __int64 *); // [rsp+50h] [rbp-40h]

  v7 = v20;
  v10 = *(_BYTE **)a7;
  v11 = *(_QWORD *)(a7 + 8);
  v21 = 0;
  v19[0] = (__int64)v20;
  if ( &v10[v11] && !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v18 = v11;
  if ( v11 > 0xF )
  {
    v19[0] = sub_22409D0(v19, &v18, 0);
    v14 = (_QWORD *)v19[0];
    v20[0] = v18;
LABEL_12:
    memcpy(v14, v10, v11);
    v11 = v18;
    v7 = (_QWORD *)v19[0];
    goto LABEL_6;
  }
  if ( v11 == 1 )
  {
    LOBYTE(v20[0]) = *v10;
    goto LABEL_6;
  }
  if ( v11 )
  {
    v14 = v20;
    goto LABEL_12;
  }
LABEL_6:
  v19[1] = v11;
  *((_BYTE *)v7 + v11) = 0;
  v21 = sub_221F8D0;
  v12 = sub_2214600(*(_QWORD *)(a1 + 16), a2, a3, a4, a5, a6, 0, v19);
  if ( v21 )
    v21((unsigned __int64 *)v19);
  return v12;
}
