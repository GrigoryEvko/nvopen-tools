// Function: sub_1FDDCF0
// Address: 0x1fddcf0
//
__int64 __fastcall sub_1FDDCF0(unsigned __int64 *a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rcx

  result = *((unsigned int *)a1 + 2);
  v2 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a1 + 8);
  v3 = *a1;
  if ( (unsigned int)result > 0x40 )
  {
    result = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
    *(_QWORD *)(v3 + 8 * result) &= v2;
  }
  else
  {
    *a1 = v3 & v2;
  }
  return result;
}
