// Function: sub_C46F20
// Address: 0xc46f20
//
__int64 __fastcall sub_C46F20(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rdx
  unsigned __int64 *v4; // rdi
  unsigned __int64 v5; // rax
  unsigned __int64 *v6; // rcx
  __int64 v7; // rdx

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(unsigned __int64 **)a1;
  if ( (unsigned int)v3 > 0x40 )
  {
    sub_C46E50(v4, a2, (unsigned __int64)(v3 + 63) >> 6);
    v3 = *(unsigned int *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = (char *)v4 - a2;
  }
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  v6 = *(unsigned __int64 **)a1;
  if ( (_DWORD)v3 )
  {
    if ( (unsigned int)v3 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      v6[v7] &= v5;
      return a1;
    }
  }
  else
  {
    v5 = 0;
  }
  *(_QWORD *)a1 = (unsigned __int64)v6 & v5;
  return a1;
}
