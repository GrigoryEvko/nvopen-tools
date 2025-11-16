// Function: sub_899F90
// Address: 0x899f90
//
__int64 __fastcall sub_899F90(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl

  v1 = sub_892240(a1);
  if ( !v1 )
    return 0;
  v2 = *(_BYTE *)(v1 + 80);
  if ( (v2 & 2) != 0 )
    return 0;
  if ( !*(_DWORD *)(*(_QWORD *)(v1 + 16) + 24LL) )
    return 1;
  if ( v2 >= 0 )
    return sub_899CC0(v1, 1, 0);
  return (unsigned __int8)v2 >> 7;
}
