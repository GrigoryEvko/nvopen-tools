// Function: sub_2217E40
// Address: 0x2217e40
//
__int64 *__fastcall sub_2217E40(__int64 *a1, __int64 a2)
{
  void *v2; // rbp
  const char *v3; // r14
  size_t v4; // rax
  size_t v5; // r13
  __int64 v7; // rax
  size_t v8[6]; // [rsp+8h] [rbp-30h] BYREF

  v2 = a1 + 2;
  v3 = *(const char **)(*(_QWORD *)(a2 + 16) + 40LL);
  *a1 = (__int64)(a1 + 2);
  if ( !v3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v4 = strlen(v3);
  v8[0] = v4;
  v5 = v4;
  if ( v4 > 0xF )
  {
    v7 = sub_22409D0(a1, v8, 0);
    *a1 = v7;
    v2 = (void *)v7;
    a1[2] = v8[0];
    goto LABEL_9;
  }
  if ( v4 != 1 )
  {
    if ( !v4 )
      goto LABEL_5;
LABEL_9:
    memcpy(v2, v3, v5);
    v4 = v8[0];
    v2 = (void *)*a1;
    goto LABEL_5;
  }
  *((_BYTE *)a1 + 16) = *v3;
LABEL_5:
  a1[1] = v4;
  *((_BYTE *)v2 + v4) = 0;
  return a1;
}
