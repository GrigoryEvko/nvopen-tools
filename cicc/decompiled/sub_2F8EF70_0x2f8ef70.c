// Function: sub_2F8EF70
// Address: 0x2f8ef70
//
__int64 __fastcall sub_2F8EF70(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8

  if ( !a2 )
    return 0;
  v2 = *(_DWORD *)(a2 + 24);
  v3 = 0;
  if ( v2 < 0 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) - 40LL * (unsigned int)~v2;
  return v3;
}
