// Function: sub_3351690
// Address: 0x3351690
//
bool __fastcall sub_3351690(_DWORD *a1)
{
  unsigned __int64 v1; // rdx
  __int64 v2; // rcx
  bool result; // al

  if ( !*(_QWORD *)a1
    || (v1 = *(unsigned int *)(*(_QWORD *)a1 + 24LL), (unsigned int)v1 > 0x31)
    || (v2 = 0x2000000001304LL, result = 1, !_bittest64(&v2, v1)) )
  {
    result = 0;
    if ( !a1[52] )
      return a1[53] != 0;
  }
  return result;
}
