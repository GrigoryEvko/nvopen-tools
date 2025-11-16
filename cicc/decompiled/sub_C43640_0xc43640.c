// Function: sub_C43640
// Address: 0xc43640
//
unsigned __int64 *__fastcall sub_C43640(unsigned __int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rcx
  __int64 v4; // rax

  v1 = *((unsigned int *)a1 + 2);
  v2 = 0xFFFFFFFFFFFFFFFFLL >> -*((_BYTE *)a1 + 8);
  v3 = *a1;
  if ( (_DWORD)v1 )
  {
    if ( (unsigned int)v1 > 0x40 )
    {
      v4 = (unsigned int)((unsigned __int64)(v1 + 63) >> 6) - 1;
      *(_QWORD *)(v3 + 8 * v4) &= v2;
      return a1;
    }
  }
  else
  {
    v2 = 0;
  }
  *a1 = v3 & v2;
  return a1;
}
