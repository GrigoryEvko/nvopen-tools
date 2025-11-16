// Function: sub_15A4AD0
// Address: 0x15a4ad0
//
__int64 __fastcall sub_15A4AD0(__int64 ***a1, __int64 a2)
{
  __int64 **v2; // rax
  __int64 v3; // rdx
  int v4; // eax

  v2 = *a1;
  if ( *((_BYTE *)*a1 + 8) == 16 )
    v2 = (__int64 **)*v2[2];
  v3 = a2;
  v4 = *((_DWORD *)v2 + 2) >> 8;
  if ( *(_BYTE *)(a2 + 8) == 16 )
    v3 = **(_QWORD **)(a2 + 16);
  if ( *(_DWORD *)(v3 + 8) >> 8 == v4 )
    return sub_15A4510(a1, (__int64 **)a2, 0);
  else
    return sub_15A3300((unsigned __int64)a1, a2, 0);
}
