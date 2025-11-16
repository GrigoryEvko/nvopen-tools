// Function: sub_9CDFE0
// Address: 0x9cdfe0
//
__int64 *__fastcall sub_9CDFE0(__int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  char v7; // [rsp+8h] [rbp-18h]

  *(_QWORD *)(a2 + 16) = (a3 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
  *(_DWORD *)(a2 + 32) = 0;
  v4 = a3 & 0x3F;
  if ( (_DWORD)v4 && (sub_9C66D0((__int64)&v6, a2, v4, a4), (v7 & 1) != 0) )
  {
    *a1 = v6 | 1;
    return a1;
  }
  else
  {
    *a1 = 1;
    return a1;
  }
}
