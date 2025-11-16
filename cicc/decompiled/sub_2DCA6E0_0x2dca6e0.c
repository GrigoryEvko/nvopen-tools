// Function: sub_2DCA6E0
// Address: 0x2dca6e0
//
__int64 __fastcall sub_2DCA6E0(__int64 a1, __int64 a2)
{
  return *(_DWORD *)(*(_QWORD *)(a1 + 312)
                   + 16LL
                   * (*(unsigned __int16 *)(*(_QWORD *)sub_2FF6500(a1, a2, 1) + 24LL)
                    + *(_DWORD *)(a1 + 328)
                    * (unsigned int)((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3))
                   + 4) >> 3;
}
