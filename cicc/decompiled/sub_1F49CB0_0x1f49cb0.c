// Function: sub_1F49CB0
// Address: 0x1f49cb0
//
const char *__fastcall sub_1F49CB0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  bool v3; // zf
  const char *result; // rax
  const char *v5; // r12

  v2 = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 24LL * a2);
  v3 = *(_QWORD *)(a1 + 72) + v2 == 0;
  result = (const char *)(*(_QWORD *)(a1 + 72) + v2);
  v5 = result;
  if ( !v3 )
  {
    strlen(result);
    return v5;
  }
  return result;
}
