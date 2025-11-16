// Function: sub_C4F0E0
// Address: 0xc4f0e0
//
unsigned __int64 __fastcall sub_C4F0E0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v3; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v1 <= 0x40 )
    return sub_C4EC70((int *)(a1 + 8), (__int64 *)a1);
  v3 = sub_C4ED70(*(__int64 **)a1, *(_QWORD *)a1 + 8 * ((unsigned __int64)(v1 + 63) >> 6));
  return sub_C4ECF0((int *)(a1 + 8), (__int64 *)&v3);
}
