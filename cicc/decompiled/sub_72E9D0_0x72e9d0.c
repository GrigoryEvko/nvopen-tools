// Function: sub_72E9D0
// Address: 0x72e9d0
//
__int64 __fastcall sub_72E9D0(_BYTE *a1, _QWORD *a2, int *a3)
{
  __int64 *v4; // rax
  __int64 v5; // rcx

  if ( a1[173] != 12 || a1[176] != 1 || (a1[177] & 0x20) == 0 )
    return 0;
  v4 = sub_72E9A0((__int64)a1);
  if ( *((_BYTE *)v4 + 24) != 1 || (v4[7] & 0xFD) != 5 || (*((_BYTE *)v4 + 58) & 0xA) != 0 )
    return 0;
  if ( *(_BYTE *)(*v4 + 140) != 14 || *(_BYTE *)(*v4 + 160) != 2 )
  {
    v5 = v4[9];
    if ( *(_BYTE *)(v5 + 24) == 2 )
    {
      *a3 = ((*((_BYTE *)v4 + 27) >> 1) ^ 1) & 1;
      *a2 = *(_QWORD *)(v5 + 56);
      return 1;
    }
    return 0;
  }
  return 0;
}
