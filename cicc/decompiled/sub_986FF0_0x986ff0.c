// Function: sub_986FF0
// Address: 0x986ff0
//
__int64 __fastcall sub_986FF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rsi
  unsigned __int64 v3; // rdx

  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
  {
    memset(*(void **)a1, -1, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
    result = *(unsigned int *)(a1 + 8);
    v2 = *(_QWORD *)a1;
  }
  else
  {
    *(_QWORD *)a1 = -1;
    v2 = -1;
  }
  v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)result;
  if ( (_DWORD)result )
  {
    if ( (unsigned int)result > 0x40 )
    {
      result = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
      *(_QWORD *)(v2 + 8 * result) &= v3;
      return result;
    }
  }
  else
  {
    v3 = 0;
  }
  *(_QWORD *)a1 = v2 & v3;
  return result;
}
