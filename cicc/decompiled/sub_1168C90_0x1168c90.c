// Function: sub_1168C90
// Address: 0x1168c90
//
void __fastcall sub_1168C90(__int64 *a1, _BYTE *a2, __int64 a3)
{
  size_t v4; // r12
  _BYTE *v5; // rdi
  __int64 v6; // rax
  size_t v7[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v4 = a3 - (_QWORD)a2;
  v7[0] = a3 - (_QWORD)a2;
  if ( (unsigned __int64)(a3 - (_QWORD)a2) > 0xF )
  {
    v6 = sub_22409D0(a1, v7, 0);
    *a1 = v6;
    v5 = (_BYTE *)v6;
    a1[2] = v7[0];
    goto LABEL_10;
  }
  v5 = (_BYTE *)*a1;
  if ( v4 != 1 )
  {
    if ( !v4 )
      goto LABEL_6;
LABEL_10:
    memcpy(v5, a2, v4);
    v4 = v7[0];
    v5 = (_BYTE *)*a1;
    goto LABEL_6;
  }
  *v5 = *a2;
  v4 = v7[0];
  v5 = (_BYTE *)*a1;
LABEL_6:
  a1[1] = v4;
  v5[v4] = 0;
}
