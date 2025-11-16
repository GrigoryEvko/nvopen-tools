// Function: sub_9DD260
// Address: 0x9dd260
//
__int64 __fastcall sub_9DD260(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  char v6; // [rsp+8h] [rbp-18h]

  sub_9DB240((__int64)&v5, a2, a3, 0);
  if ( (v6 & 1) != 0 )
  {
    v4 = v5;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v4 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = v5;
  }
  return a1;
}
