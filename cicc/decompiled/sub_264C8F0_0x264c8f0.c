// Function: sub_264C8F0
// Address: 0x264c8f0
//
__int64 __fastcall sub_264C8F0(
        __int64 a1,
        int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        int *a9,
        int *a10)
{
  int *v10; // rbx
  __int64 result; // rax
  _BYTE v12[96]; // [rsp+0h] [rbp-60h] BYREF

  v10 = a9;
  while ( a2 != v10 )
  {
    result = sub_22B6470((__int64)v12, a1, v10);
    do
      ++v10;
    while ( v10 != a10 && (unsigned int)*v10 > 0xFFFFFFFD );
  }
  return result;
}
