// Function: sub_16A7DC0
// Address: 0x16a7dc0
//
__int64 __fastcall sub_16A7DC0(__int64 *a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx

  sub_16A7D00((_QWORD *)*a1, ((unsigned __int64)*((unsigned int *)a1 + 2) + 63) >> 6, a2);
  result = *((unsigned int *)a1 + 2);
  v3 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a1 + 8);
  if ( (unsigned int)result > 0x40 )
  {
    v4 = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
    result = *a1;
    *(_QWORD *)(*a1 + 8 * v4) &= v3;
  }
  else
  {
    *a1 &= v3;
  }
  return result;
}
