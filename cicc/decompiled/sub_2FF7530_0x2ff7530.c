// Function: sub_2FF7530
// Address: 0x2ff7530
//
__int64 __fastcall sub_2FF7530(__int64 a1, int a2)
{
  return *(unsigned __int16 *)(*(_QWORD *)(a1 + 264)
                             + 4LL * (unsigned int)(*(_DWORD *)(a1 + 96) * *(_DWORD *)(a1 + 328) + a2)
                             + 2);
}
