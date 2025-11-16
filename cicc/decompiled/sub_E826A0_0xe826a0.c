// Function: sub_E826A0
// Address: 0xe826a0
//
const char *__fastcall sub_E826A0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  bool v4; // zf
  const char *result; // rax
  const char *v6; // r12

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(unsigned int *)(*(_QWORD *)(v2 + 8) + 4LL * a2);
  v4 = *(_QWORD *)(v2 + 16) + v3 == 0;
  result = (const char *)(*(_QWORD *)(v2 + 16) + v3);
  v6 = result;
  if ( !v4 )
  {
    strlen(result);
    return v6;
  }
  return result;
}
