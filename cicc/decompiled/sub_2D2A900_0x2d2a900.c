// Function: sub_2D2A900
// Address: 0x2d2a900
//
__int64 __fastcall sub_2D2A900(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)result )
  {
    sub_2D2A3E0(a1, (char *)sub_2D227B0, 0, a4, a5, a6);
    result = 0;
    *(_DWORD *)(a1 + 192) = 0;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 184) = 0;
    memset(
      (void *)((a1 + 8) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)a1 - (((_DWORD)a1 + 8) & 0xFFFFFFF8) + 192) >> 3));
  }
  *(_DWORD *)(a1 + 196) = 0;
  return result;
}
