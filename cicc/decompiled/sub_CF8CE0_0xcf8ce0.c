// Function: sub_CF8CE0
// Address: 0xcf8ce0
//
unsigned __int64 __fastcall sub_CF8CE0(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned int v4; // eax

  v1 = a1[3];
  if ( *(_BYTE *)v1 != 85 )
    return 0;
  v2 = *(_QWORD *)(v1 - 32);
  if ( !v2
    || *(_BYTE *)v2
    || *(_QWORD *)(v2 + 24) != *(_QWORD *)(v1 + 80)
    || *(_DWORD *)(v2 + 36) != 11
    || *a1 == *(_QWORD *)(v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF)) )
  {
    return 0;
  }
  v4 = sub_BD2910((__int64)a1);
  return sub_B49810(v1, v4);
}
