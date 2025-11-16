// Function: sub_16A9900
// Address: 0x16a9900
//
__int64 __fastcall sub_16A9900(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  __int64 result; // rax

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *a2;
  v4 = *(_QWORD *)a1;
  if ( (unsigned int)v2 > 0x40 )
    return sub_16A98D0(v4, v3, (unsigned __int64)(v2 + 63) >> 6);
  result = v4 > v3;
  if ( v4 < v3 )
    return 0xFFFFFFFFLL;
  return result;
}
