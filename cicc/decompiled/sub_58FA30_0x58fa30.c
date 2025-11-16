// Function: sub_58FA30
// Address: 0x58fa30
//
size_t __fastcall sub_58FA30(__int64 *a1, const char *a2)
{
  void *v2; // r14
  size_t v3; // rax
  size_t v4; // r12
  size_t result; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  size_t v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v3 = strlen(a2);
  v8[0] = v3;
  v4 = v3;
  if ( v3 > 0xF )
  {
    v7 = sub_22409D0(a1, v8, 0);
    *a1 = v7;
    v2 = (void *)v7;
    a1[2] = v8[0];
    goto LABEL_9;
  }
  if ( v3 != 1 )
  {
    if ( !v3 )
      goto LABEL_5;
LABEL_9:
    memcpy(v2, a2, v4);
    goto LABEL_5;
  }
  *((_BYTE *)a1 + 16) = *a2;
LABEL_5:
  result = v8[0];
  v6 = *a1;
  a1[1] = v8[0];
  *(_BYTE *)(v6 + result) = 0;
  return result;
}
