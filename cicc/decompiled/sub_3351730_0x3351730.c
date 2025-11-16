// Function: sub_3351730
// Address: 0x3351730
//
__int64 __fastcall sub_3351730(__int64 a1, unsigned int *a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 result; // rax

  if ( !*(_QWORD *)a2
    || (v2 = *(unsigned int *)(*(_QWORD *)a2 + 24LL), (unsigned int)v2 > 0x31)
    || (v3 = 0x2000000001304LL, result = 0, !_bittest64(&v3, v2)) )
  {
    result = a2[52];
    if ( a2[53] )
    {
      if ( !(_DWORD)result )
        return result;
    }
    else if ( (_DWORD)result )
    {
      return 0xFFFF;
    }
    return *(unsigned int *)(*(_QWORD *)(a1 + 96) + 4LL * a2[50]);
  }
  return result;
}
