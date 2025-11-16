// Function: sub_23088C0
// Address: 0x23088c0
//
__int64 __fastcall sub_23088C0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  const void *v5; // r15
  const char *v6; // rax
  size_t v7; // rdx

  v2 = **a1;
  v3 = v2 + 32LL * *((unsigned int *)*a1 + 2);
  if ( v2 == v3 )
    return 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v2 + 8);
    v5 = *(const void **)v2;
    v6 = sub_BD5D20(a2);
    if ( v7 == v4 && (!v7 || !memcmp(v6, v5, v7)) )
      break;
    v2 += 32;
    if ( v3 == v2 )
      return 0;
  }
  return 1;
}
