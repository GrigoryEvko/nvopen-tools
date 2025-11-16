// Function: sub_36DA3C0
// Address: 0x36da3c0
//
__int64 *__fastcall sub_36DA3C0(__int64 *a1, __int64 a2)
{
  void *v2; // r15
  __int64 v3; // r13
  const char *v4; // rax
  const char *v5; // r14
  size_t v6; // rax
  size_t v7; // rbx
  __int64 v9; // rax
  size_t v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 2;
  v3 = sub_3936750();
  sub_314BE80(a2, v3);
  v4 = (const char *)sub_3936860(v3, 1);
  *a1 = (__int64)(a1 + 2);
  if ( !v4 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v5 = v4;
  v6 = strlen(v4);
  v10[0] = v6;
  v7 = v6;
  if ( v6 > 0xF )
  {
    v9 = sub_22409D0((__int64)a1, v10, 0);
    *a1 = v9;
    v2 = (void *)v9;
    a1[2] = v10[0];
    goto LABEL_9;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
      goto LABEL_5;
LABEL_9:
    memcpy(v2, v5, v7);
    v6 = v10[0];
    v2 = (void *)*a1;
    goto LABEL_5;
  }
  *((_BYTE *)a1 + 16) = *v5;
LABEL_5:
  a1[1] = v6;
  *((_BYTE *)v2 + v6) = 0;
  sub_39367A0(v3);
  return a1;
}
