// Function: sub_2EF3070
// Address: 0x2ef3070
//
unsigned __int64 __fastcall sub_2EF3070(__int64 a1, int a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx

  result = 0;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF));
  if ( v3 )
  {
    if ( (v3 & 4) == 0 )
      return v3 & 0xFFFFFFFFFFFFFFF8LL;
  }
  return result;
}
