// Function: sub_1D00FE0
// Address: 0x1d00fe0
//
bool __fastcall sub_1D00FE0(_DWORD *a1)
{
  unsigned __int64 v1; // rdx
  __int64 v2; // rcx
  bool result; // al

  if ( !*(_QWORD *)a1
    || (v1 = *(unsigned __int16 *)(*(_QWORD *)a1 + 24LL), (unsigned __int16)v1 > 0x2Eu)
    || (v2 = 0x400000000584LL, result = 1, !_bittest64(&v2, v1)) )
  {
    result = 0;
    if ( !a1[50] )
      return a1[51] != 0;
  }
  return result;
}
