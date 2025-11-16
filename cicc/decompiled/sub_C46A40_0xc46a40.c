// Function: sub_C46A40
// Address: 0xc46a40
//
__int64 __fastcall sub_C46A40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // rdx

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD **)a1;
  if ( (unsigned int)v3 > 0x40 )
  {
    sub_C46200(v4, a2, (unsigned __int64)(v3 + 63) >> 6);
    v3 = *(unsigned int *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = (char *)v4 + a2;
  }
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  v6 = *(_QWORD *)a1;
  if ( (_DWORD)v3 )
  {
    if ( (unsigned int)v3 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      *(_QWORD *)(v6 + 8 * v7) &= v5;
      return a1;
    }
  }
  else
  {
    v5 = 0;
  }
  *(_QWORD *)a1 = v6 & v5;
  return a1;
}
