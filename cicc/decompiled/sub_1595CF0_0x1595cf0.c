// Function: sub_1595CF0
// Address: 0x1595cf0
//
__int64 __fastcall sub_1595CF0(__int64 a1)
{
  char *v1; // rbx
  unsigned int v2; // r14d
  unsigned int v3; // r13d
  size_t v4; // r12
  int v5; // r15d
  int v7; // [rsp+Ch] [rbp-34h]

  v1 = (char *)sub_1595920(a1);
  v2 = sub_1595900(a1);
  v3 = v2;
  v7 = sub_15958F0(a1);
  if ( v7 == 1 )
    return 1;
  v4 = v2;
  v5 = 1;
  while ( !memcmp(v1, &v1[v2], v4) )
  {
    ++v5;
    v2 += v3;
    if ( v7 == v5 )
      return 1;
  }
  return 0;
}
