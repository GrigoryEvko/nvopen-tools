// Function: sub_1ED8200
// Address: 0x1ed8200
//
__int64 __fastcall sub_1ED8200(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // edx
  char v4; // al
  __int64 v6; // rdi
  __int64 v7; // rsi

  v2 = a1[2];
  v3 = a2[2];
  if ( v2 != v3 )
    return v3 < v2 ? -1 : 1;
  v4 = *((_BYTE *)a1 + 12);
  if ( v4 != *((_BYTE *)a2 + 12) )
    return v4 == 0 ? 1 : -1;
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)a2;
  v2 = ((__int64)(*(_QWORD *)(v6 + 72) - *(_QWORD *)(v6 + 64)) >> 3)
     + ((__int64)(*(_QWORD *)(v6 + 96) - *(_QWORD *)(v6 + 88)) >> 3);
  v3 = ((__int64)(*(_QWORD *)(v7 + 72) - *(_QWORD *)(v7 + 64)) >> 3)
     + ((__int64)(*(_QWORD *)(v7 + 96) - *(_QWORD *)(v7 + 88)) >> 3);
  if ( v2 == v3 )
    return 2 * (unsigned int)(*(_DWORD *)(v6 + 48) >= *(_DWORD *)(v7 + 48)) - 1;
  else
    return v3 < v2 ? -1 : 1;
}
