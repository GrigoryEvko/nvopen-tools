// Function: sub_866940
// Address: 0x866940
//
__int64 __fastcall sub_866940(int a1, _DWORD *a2, __int64 *a3, _DWORD *a4, _DWORD *a5)
{
  __int64 result; // rax

  if ( a1
    && (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
    && dword_4F04C64 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0 )
  {
    *a2 = 1;
    sub_866880(a3, (__int64)a2);
  }
  else
  {
    *a2 = 0;
    *a3 = 0;
  }
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *a4 = (*(_BYTE *)(result + 12) & 4) != 0;
  *(_BYTE *)(result + 12) |= 4u;
  *a5 = dword_4F04C3C;
  *(_BYTE *)(result + 7) |= 0x10u;
  dword_4F04C3C = 1;
  return result;
}
