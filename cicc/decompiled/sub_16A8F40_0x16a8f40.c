// Function: sub_16A8F40
// Address: 0x16a8f40
//
__int64 __fastcall sub_16A8F40(__int64 *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx

  sub_16A8F20((_QWORD *)*a1, ((unsigned __int64)*((unsigned int *)a1 + 2) + 63) >> 6);
  result = *((unsigned int *)a1 + 2);
  v2 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a1 + 8);
  if ( (unsigned int)result > 0x40 )
  {
    v3 = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
    result = *a1;
    *(_QWORD *)(*a1 + 8 * v3) &= v2;
  }
  else
  {
    *a1 &= v2;
  }
  return result;
}
