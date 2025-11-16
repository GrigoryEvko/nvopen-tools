// Function: sub_C43D10
// Address: 0xc43d10
//
unsigned __int64 __fastcall sub_C43D10(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rdx
  unsigned __int64 v3; // rcx
  __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  __int64 v7; // rdx

  result = *(_QWORD *)a1;
  v2 = *(unsigned int *)(a1 + 8);
  v3 = (unsigned __int64)(v2 + 63) >> 6;
  if ( v3 )
  {
    v4 = result + 8LL * (unsigned int)(v3 - 1) + 8;
    do
    {
      *(_QWORD *)result = ~*(_QWORD *)result;
      result += 8LL;
    }
    while ( result != v4 );
    v2 = *(unsigned int *)(a1 + 8);
    result = *(_QWORD *)a1;
  }
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
  v6 = v5;
  if ( (_DWORD)v2 )
  {
    if ( (unsigned int)v2 > 0x40 )
    {
      v7 = (unsigned int)((unsigned __int64)(v2 + 63) >> 6) - 1;
      *(_QWORD *)(result + 8 * v7) &= v5;
      return result;
    }
  }
  else
  {
    v6 = 0;
  }
  result &= v6;
  *(_QWORD *)a1 = result;
  return result;
}
