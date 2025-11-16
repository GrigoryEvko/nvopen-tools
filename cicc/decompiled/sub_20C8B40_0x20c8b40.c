// Function: sub_20C8B40
// Address: 0x20c8b40
//
__int64 __fastcall sub_20C8B40(__int64 a1, __int64 a2, __int64 a3, const char **a4)
{
  if ( a3
    && (*(_DWORD *)(a3 + 20) & 0xFFFFFFF) != 0
    && *(_BYTE *)(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)) + 16LL) != 9 )
  {
    return sub_20C84C0(a1, a2, a3, a4);
  }
  else
  {
    return 1;
  }
}
