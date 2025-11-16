// Function: sub_C46B40
// Address: 0xc46b40
//
__int64 __fastcall sub_C46B40(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *a2;
  v5 = *(_QWORD *)a1;
  if ( (unsigned int)v3 > 0x40 )
  {
    sub_C46AD0(v5, v4, 0, ((unsigned __int64)(unsigned int)v3 + 63) >> 6);
    v3 = *(unsigned int *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = v5 - v4;
  }
  v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  v7 = *(_QWORD *)a1;
  if ( (_DWORD)v3 )
  {
    if ( (unsigned int)v3 > 0x40 )
    {
      v8 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      *(_QWORD *)(v7 + 8 * v8) &= v6;
      return a1;
    }
  }
  else
  {
    v6 = 0;
  }
  *(_QWORD *)a1 = v7 & v6;
  return a1;
}
