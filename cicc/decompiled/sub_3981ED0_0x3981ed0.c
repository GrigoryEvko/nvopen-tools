// Function: sub_3981ED0
// Address: 0x3981ed0
//
unsigned __int64 __fastcall sub_3981ED0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rdx

  result = sub_3981E80(a1);
  if ( result )
  {
    v2 = *(_QWORD *)(result + 40);
    result = v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v2 & 4) == 0 )
      return 0;
  }
  return result;
}
