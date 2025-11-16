// Function: sub_370D250
// Address: 0x370d250
//
__int64 *__fastcall sub_370D250(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  unsigned __int64 v6; // rax
  __int64 v7[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = a2 + 16;
  if ( *(_QWORD *)(a2 + 56) && !*(_QWORD *)(a2 + 72) && !*(_QWORD *)(a2 + 64) )
  {
    sub_3700FD0(v7, v3);
    v6 = v7[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v7[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = 0;
      v7[0] = v6 | 1;
      sub_9C6670(a1, v7);
      sub_9C66B0(v7);
      return a1;
    }
    v7[0] = 0;
    sub_9C66B0(v7);
  }
  if ( *(_BYTE *)(a2 + 14) )
    *(_BYTE *)(a2 + 14) = 0;
  sub_3700E20(v7, v3);
  v4 = v7[0] | 1;
  if ( (v7[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v4 = 1;
  *a1 = v4;
  return a1;
}
