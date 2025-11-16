// Function: sub_33CC5F0
// Address: 0x33cc5f0
//
char __fastcall sub_33CC5F0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *(unsigned int *)(a1[1] + 556);
  if ( (unsigned int)v2 <= 0x1F && (v3 = 3623879202LL, _bittest64(&v3, v2)) )
    return sub_B2D610(*a1, 18);
  else
    return sub_33CC5C0(a2);
}
