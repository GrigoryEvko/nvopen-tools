// Function: sub_2509800
// Address: 0x2509800
//
__int64 __fastcall sub_2509800(_QWORD *a1)
{
  __int64 v1; // rdx
  unsigned int v2; // r8d
  unsigned __int8 *v3; // rax
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rcx

  v1 = *a1 & 3LL;
  if ( v1 == 3 )
    return 7;
  v2 = 1;
  if ( v1 == 2 )
    return v2;
  v2 = 0;
  v3 = (unsigned __int8 *)(*a1 & 0xFFFFFFFFFFFFFFFCLL);
  if ( !v3 )
    return v2;
  v4 = *v3;
  if ( (_BYTE)v4 == 22 )
    return 6;
  if ( (_BYTE)v4 )
  {
    v2 = 1;
    if ( (unsigned __int8)v4 > 0x1Cu )
    {
      v5 = (unsigned int)(v4 - 34);
      if ( (unsigned __int8)v5 <= 0x33u )
      {
        v6 = 0x8000000000041LL;
        if ( _bittest64(&v6, v5) )
        {
          LOBYTE(v2) = (_BYTE)v1 != 1;
          return 2 * v2 + 3;
        }
      }
    }
    return v2;
  }
  LOBYTE(v2) = (_BYTE)v1 != 1;
  return 2 * v2 + 2;
}
