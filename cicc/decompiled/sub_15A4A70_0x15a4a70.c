// Function: sub_15A4A70
// Address: 0x15a4a70
//
__int64 __fastcall sub_15A4A70(__int64 ***a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rcx
  __int64 **v4; // rdx

  v2 = *(_BYTE *)(a2 + 8);
  v3 = a2;
  if ( v2 == 16 )
  {
    v3 = **(_QWORD **)(a2 + 16);
    v2 = *(_BYTE *)(v3 + 8);
  }
  if ( v2 == 11 )
    return sub_15A4180((unsigned __int64)a1, (__int64 **)a2, 0);
  v4 = *a1;
  if ( *((_BYTE *)*a1 + 8) == 16 )
    v4 = (__int64 **)*v4[2];
  if ( v2 == 15 && *((_DWORD *)v4 + 2) >> 8 != *(_DWORD *)(v3 + 8) >> 8 )
    return sub_15A3300((unsigned __int64)a1, a2, 0);
  else
    return sub_15A4510(a1, (__int64 **)a2, 0);
}
