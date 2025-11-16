// Function: sub_22BDBB0
// Address: 0x22bdbb0
//
__int64 __fastcall sub_22BDBB0(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v2 = a1[2];
  result = a2[2];
  if ( v2 != result )
  {
    if ( v2 != -4096 && v2 != 0 && v2 != -8192 )
    {
      sub_BD60C0(a1);
      result = a2[2];
    }
    a1[2] = result;
    if ( result != 0 && result != -4096 && result != -8192 )
      return sub_BD6050(a1, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  }
  return result;
}
