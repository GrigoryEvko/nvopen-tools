// Function: sub_C47170
// Address: 0xc47170
//
__int64 __fastcall sub_C47170(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rcx

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  if ( (unsigned int)v3 > 0x40 )
  {
    sub_C46FF0(v4, v4, a2, 0, (unsigned __int64)(v3 + 63) >> 6, (unsigned __int64)(v3 + 63) >> 6, 0);
    v3 = *(unsigned int *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = v4 * a2;
  }
  v5 = *(_QWORD *)a1;
  v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  if ( (_DWORD)v3 )
  {
    if ( (unsigned int)v3 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      *(_QWORD *)(v5 + 8 * v7) &= v6;
      return a1;
    }
  }
  else
  {
    v6 = 0;
  }
  *(_QWORD *)a1 = v5 & v6;
  return a1;
}
