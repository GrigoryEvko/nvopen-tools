// Function: sub_2E50190
// Address: 0x2e50190
//
__int64 __fastcall sub_2E50190(__int64 a1, char a2, __int64 a3)
{
  int v3; // eax

  if ( (_DWORD)a3 && (v3 = *(_DWORD *)(a1 + 44), (v3 & 4) == 0) && (v3 & 8) != 0 )
    return sub_2E88A90(a1, 1LL << a2, a3);
  else
    return (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> a2) & 1LL;
}
