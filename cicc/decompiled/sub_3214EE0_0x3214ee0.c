// Function: sub_3214EE0
// Address: 0x3214ee0
//
unsigned __int64 __fastcall sub_3214EE0(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 40);
  result = 0;
  if ( v1 )
  {
    if ( (v1 & 4) == 0 )
      return v1 & 0xFFFFFFFFFFFFFFF8LL;
  }
  return result;
}
