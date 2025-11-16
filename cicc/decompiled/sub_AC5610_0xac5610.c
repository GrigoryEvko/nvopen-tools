// Function: sub_AC5610
// Address: 0xac5610
//
__int64 __fastcall sub_AC5610(__int64 a1)
{
  char *v1; // rbx
  unsigned int v2; // r14d
  unsigned int v3; // r13d
  size_t v4; // r12
  int v5; // r15d
  int v7; // [rsp+Ch] [rbp-34h]

  v1 = (char *)sub_AC52D0(a1);
  v2 = sub_AC52A0(a1);
  v3 = v2;
  v7 = sub_AC5290(a1);
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
