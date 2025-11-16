// Function: sub_C46E90
// Address: 0xc46e90
//
__int64 __fastcall sub_C46E90(__int64 a1)
{
  __int64 v2; // rdx
  unsigned __int64 *v3; // rdi
  unsigned __int64 v4; // rax
  unsigned __int64 *v5; // rcx
  __int64 v6; // rdx

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(unsigned __int64 **)a1;
  if ( (unsigned int)v2 > 0x40 )
  {
    sub_C46E50(v3, 1u, (unsigned __int64)(v2 + 63) >> 6);
    v2 = *(unsigned int *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = (char *)v3 - 1;
  }
  v4 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
  v5 = *(unsigned __int64 **)a1;
  if ( (_DWORD)v2 )
  {
    if ( (unsigned int)v2 > 0x40 )
    {
      v6 = (unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1;
      v5[v6] &= v4;
      return a1;
    }
  }
  else
  {
    v4 = 0;
  }
  *(_QWORD *)a1 = (unsigned __int64)v5 & v4;
  return a1;
}
