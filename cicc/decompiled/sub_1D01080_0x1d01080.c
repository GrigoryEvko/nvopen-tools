// Function: sub_1D01080
// Address: 0x1d01080
//
__int64 __fastcall sub_1D01080(__int64 a1, unsigned int *a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 result; // rax

  if ( !*(_QWORD *)a2
    || (v2 = *(unsigned __int16 *)(*(_QWORD *)a2 + 24LL), (unsigned __int16)v2 > 0x2Eu)
    || (v3 = 0x400000000584LL, result = 0, !_bittest64(&v3, v2)) )
  {
    result = a2[50];
    if ( a2[51] )
    {
      if ( !(_DWORD)result )
        return result;
    }
    else if ( (_DWORD)result )
    {
      return 0xFFFF;
    }
    return *(unsigned int *)(*(_QWORD *)(a1 + 96) + 4LL * a2[48]);
  }
  return result;
}
