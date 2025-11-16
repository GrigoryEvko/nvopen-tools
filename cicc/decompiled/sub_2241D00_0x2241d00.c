// Function: sub_2241D00
// Address: 0x2241d00
//
__int64 *__fastcall sub_2241D00(__int64 *a1, __int64 a2, int a3)
{
  void *v3; // r13
  char *v4; // rax
  char *v5; // rbp
  size_t v6; // rax
  size_t v7; // r14
  __int64 v9; // rax
  size_t v10[6]; // [rsp+8h] [rbp-30h] BYREF

  v3 = a1 + 2;
  v4 = strerror(a3);
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
    v3 = (void *)v9;
    a1[2] = v10[0];
    goto LABEL_9;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
      goto LABEL_5;
LABEL_9:
    memcpy(v3, v5, v7);
    v6 = v10[0];
    v3 = (void *)*a1;
    goto LABEL_5;
  }
  *((_BYTE *)a1 + 16) = *v5;
LABEL_5:
  a1[1] = v6;
  *((_BYTE *)v3 + v6) = 0;
  return a1;
}
