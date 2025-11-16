// Function: sub_A01BF0
// Address: 0xa01bf0
//
__int64 __fastcall sub_A01BF0(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdx

  result = sub_A14C90(a1, a2, a3, a4, 0);
  if ( !result && a2 < (unsigned int)((__int64)(a1[1] - *a1) >> 5) )
  {
    v6 = *(_QWORD *)(*a1 + 32LL * a2 + 16);
    if ( v6 )
    {
      if ( a3 == *(_QWORD *)(v6 + 8) )
        return sub_ACADE0(a3);
    }
  }
  return result;
}
