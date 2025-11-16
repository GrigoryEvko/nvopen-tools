// Function: sub_222F100
// Address: 0x222f100
//
void **__fastcall sub_222F100(void **a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rbp
  size_t v4; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(_BYTE **)(v2 + 56);
  if ( !v3 )
    sub_426248((__int64)"basic_string::_S_construct null not valid");
  v4 = strlen(*(const char **)(v2 + 56));
  if ( v3 == &v3[v4] )
    *a1 = &unk_4FD67D8;
  else
    *a1 = sub_222EC60(v3, (__int64)&v3[v4]);
  return a1;
}
