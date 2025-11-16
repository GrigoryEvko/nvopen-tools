// Function: sub_AF4EB0
// Address: 0xaf4eb0
//
unsigned __int64 __fastcall sub_AF4EB0(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r13
  unsigned __int64 *i; // [rsp+8h] [rbp-28h] BYREF

  v1 = 0;
  v2 = *(unsigned __int64 **)(a1 + 16);
  v3 = *(unsigned __int64 **)(a1 + 24);
  for ( i = v2; v3 != v2; i = v2 )
  {
    if ( *v2 == 4101 && v1 < v2[1] + 1 )
      v1 = v2[1] + 1;
    v2 += (unsigned int)sub_AF4160(&i);
  }
  return v1;
}
