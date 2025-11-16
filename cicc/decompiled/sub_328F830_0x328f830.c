// Function: sub_328F830
// Address: 0x328f830
//
__int64 __fastcall sub_328F830(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        int a7)
{
  __int64 v8; // rax
  __int64 result; // rax
  int v10[3]; // [rsp+Ch] [rbp-14h] BYREF

  v8 = *a1;
  v10[0] = 2;
  result = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD, _QWORD, int *))(v8 + 2264))(
             a1,
             a2,
             a3,
             a4,
             a5,
             a6,
             v10);
  if ( result )
  {
    if ( v10[0] <= a7 )
      return result;
    if ( !*(_QWORD *)(result + 56) )
      sub_33ECEA0(a4, result);
  }
  return 0;
}
