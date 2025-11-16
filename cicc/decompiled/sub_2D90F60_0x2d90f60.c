// Function: sub_2D90F60
// Address: 0x2d90f60
//
__int64 *__fastcall sub_2D90F60(__int64 *a1)
{
  void *v2; // rdi
  __int64 v3; // rax
  _BYTE *v4; // r14
  size_t v5; // r13
  __int64 v7; // rax
  size_t v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 2;
  v3 = qword_501CE58;
  *a1 = (__int64)v2;
  v4 = *(_BYTE **)(v3 + 136);
  v5 = *(_QWORD *)(v3 + 144);
  if ( &v4[v5] && !v4 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v8[0] = *(_QWORD *)(v3 + 144);
  if ( v5 > 0xF )
  {
    v7 = sub_22409D0((__int64)a1, v8, 0);
    *a1 = v7;
    v2 = (void *)v7;
    a1[2] = v8[0];
    goto LABEL_10;
  }
  if ( v5 != 1 )
  {
    if ( !v5 )
      goto LABEL_6;
LABEL_10:
    memcpy(v2, v4, v5);
    v5 = v8[0];
    v2 = (void *)*a1;
    goto LABEL_6;
  }
  *((_BYTE *)a1 + 16) = *v4;
LABEL_6:
  a1[1] = v5;
  *((_BYTE *)v2 + v5) = 0;
  return a1;
}
