// Function: sub_2F60F30
// Address: 0x2f60f30
//
__int64 __fastcall sub_2F60F30(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // edx
  char v4; // al
  _DWORD *v6; // rdi

  v2 = a1[2];
  v3 = a2[2];
  if ( v2 != v3 )
    return v3 < v2 ? -1 : 1;
  v4 = *((_BYTE *)a1 + 12);
  if ( v4 != *((_BYTE *)a2 + 12) )
    return v4 == 0 ? 1 : -1;
  v6 = *(_DWORD **)a1;
  v2 = v6[18] + v6[30];
  v3 = *(_DWORD *)(*(_QWORD *)a2 + 72LL) + *(_DWORD *)(*(_QWORD *)a2 + 120LL);
  if ( v2 == v3 )
    return 2 * (unsigned int)(v6[6] >= *(_DWORD *)(*(_QWORD *)a2 + 24LL)) - 1;
  else
    return v3 < v2 ? -1 : 1;
}
