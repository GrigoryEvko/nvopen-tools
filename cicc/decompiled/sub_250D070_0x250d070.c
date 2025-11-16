// Function: sub_250D070
// Address: 0x250d070
//
unsigned __int64 __fastcall sub_250D070(_QWORD *a1)
{
  char v1; // si
  unsigned __int64 result; // rax
  __int64 v3; // rcx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax

  v1 = sub_2509800(a1);
  result = *a1 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v1 == 6 )
  {
    LODWORD(v3) = *(_DWORD *)(result + 32);
  }
  else
  {
    if ( v1 != 7 )
      goto LABEL_13;
    v3 = (__int64)(result - (*(_QWORD *)(result + 24) - 32LL * (*(_DWORD *)(*(_QWORD *)(result + 24) + 4LL) & 0x7FFFFFF))) >> 5;
  }
  if ( (int)v3 < 0 )
  {
LABEL_13:
    if ( (*a1 & 3LL) == 3 )
      return *(_QWORD *)(result + 24);
    return result;
  }
  if ( (*a1 & 3LL) == 3 )
  {
    v4 = *(_QWORD *)(result + 24);
    if ( *(_BYTE *)v4 == 22 )
      return *(_QWORD *)(result + 24);
  }
  else
  {
    v4 = *a1 & 0xFFFFFFFFFFFFFFFCLL;
    if ( *(_BYTE *)result == 22 )
      return result;
  }
  if ( v1 == 6 )
    v5 = *(unsigned int *)(result + 32);
  else
    v5 = (unsigned int)((__int64)(result
                                - (*(_QWORD *)(result + 24)
                                 - 32LL * (*(_DWORD *)(*(_QWORD *)(result + 24) + 4LL) & 0x7FFFFFF))) >> 5);
  return *(_QWORD *)(v4 + 32 * (v5 - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
}
