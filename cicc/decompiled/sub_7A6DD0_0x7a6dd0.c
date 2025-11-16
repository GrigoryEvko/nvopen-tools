// Function: sub_7A6DD0
// Address: 0x7a6dd0
//
__int64 __fastcall sub_7A6DD0(_QWORD *a1, unsigned __int64 *a2, __int64 a3)
{
  int v3; // r10d
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // r9
  unsigned int v7; // r8d

  LOBYTE(v3) = unk_4F06AC0 < *a1;
  if ( !a3 )
    return unk_4F06AC0 >= *a1;
  v5 = *a2;
  if ( *a2 > ~a3 )
  {
    v3 = 1;
    v6 = dword_4F06BA0;
    if ( v5 < dword_4F06BA0 )
      return v3 ^ 1u;
  }
  else
  {
    v5 += a3;
    v3 = (unsigned __int8)v3;
    *a2 = v5;
    v6 = dword_4F06BA0;
    if ( v5 < dword_4F06BA0 )
      return v3 ^ 1u;
  }
  v7 = 0;
  if ( *a1 <= unk_4F06AC0 - v5 / v6 )
  {
    *a1 += v5 / v6;
    v7 = v3 ^ 1;
    v5 = *a2;
    v6 = dword_4F06BA0;
  }
  *a2 = v5 % v6;
  return v7;
}
