// Function: sub_34A2530
// Address: 0x34a2530
//
__int64 __fastcall sub_34A2530(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __int64 result; // rax

  result = a1[48];
  if ( (_DWORD)result )
  {
    sub_34A2010((__int64)a1, (char *)sub_349D600, 0, a4, a5, a6);
    result = 0;
    a1[48] = 0;
    memset(a1, 0, 0xC0u);
  }
  a1[49] = 0;
  return result;
}
