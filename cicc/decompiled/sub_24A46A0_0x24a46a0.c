// Function: sub_24A46A0
// Address: 0x24a46a0
//
size_t __fastcall sub_24A46A0(__int64 *a1, const char *a2)
{
  void *v2; // r13
  size_t result; // rax
  size_t v4; // r14
  __int64 v5; // rax
  size_t v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  result = strlen(a2);
  v6[0] = result;
  v4 = result;
  if ( result > 0xF )
  {
    v5 = sub_22409D0((__int64)a1, v6, 0);
    *a1 = v5;
    v2 = (void *)v5;
    a1[2] = v6[0];
    goto LABEL_9;
  }
  if ( result != 1 )
  {
    if ( !result )
      goto LABEL_5;
LABEL_9:
    memcpy(v2, a2, v4);
    result = v6[0];
    v2 = (void *)*a1;
    goto LABEL_5;
  }
  *((_BYTE *)a1 + 16) = *a2;
LABEL_5:
  a1[1] = result;
  *((_BYTE *)v2 + result) = 0;
  return result;
}
