// Function: sub_24DE9B0
// Address: 0x24de9b0
//
__int64 __fastcall sub_24DE9B0(unsigned __int64 a1, unsigned int *a2)
{
  unsigned __int8 **v3; // r12
  unsigned __int8 **v4; // rbx
  __int64 **v5; // rsi
  __int64 v6; // rax
  __int64 result; // rax
  unsigned __int8 **i; // r13
  unsigned __int8 *v9; // rdi

  v3 = (unsigned __int8 **)a1;
  v4 = *(unsigned __int8 ***)a2;
  v5 = *(__int64 ***)(**(_QWORD **)a2 + 8LL);
  if ( v5 != *(__int64 ***)(a1 + 8) )
  {
    v6 = sub_AD4C90(a1, v5, 0);
    v4 = *(unsigned __int8 ***)a2;
    v3 = (unsigned __int8 **)v6;
  }
  result = a2[2];
  for ( i = &v4[result]; i != v4; result = sub_1021A90(v9, v3, 0, 0, 0, 0) )
    v9 = *v4++;
  return result;
}
