// Function: sub_1DF81C0
// Address: 0x1df81c0
//
__int64 __fastcall sub_1DF81C0(__int64 a1, char a2, __int64 a3)
{
  __int16 v3; // ax

  if ( (_DWORD)a3 && (v3 = *(_WORD *)(a1 + 46), (v3 & 4) == 0) && (v3 & 8) != 0 )
    return sub_1E15D00(a1, 1LL << a2, a3);
  else
    return (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> a2) & 1LL;
}
