// Function: sub_13C12E0
// Address: 0x13c12e0
//
__int64 __fastcall sub_13C12E0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 result; // rax
  __int64 *v8; // rbx
  __int64 *v9; // rax
  __int64 v10; // rdx
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (a2 & 4) != 0 )
  {
    if ( v3 < 0 )
    {
      v4 = sub_1648A40(v2);
      v6 = v4 + v5;
      if ( *(char *)(v2 + 23) < 0 )
        v6 -= sub_1648A40(v2);
      if ( (unsigned int)(v6 >> 4) )
        return 63;
    }
    v8 = (__int64 *)(v2 - 24);
  }
  else
  {
    if ( v3 < 0 )
    {
      v12 = sub_1648A40(v2);
      v14 = v12 + v13;
      if ( *(char *)(v2 + 23) < 0 )
        v14 -= sub_1648A40(v2);
      if ( (unsigned int)(v14 >> 4) )
        return 63;
    }
    v8 = (__int64 *)(v2 - 72);
  }
  if ( *(_BYTE *)(*v8 + 16) )
    return 63;
  v9 = sub_13C1210(a1, *v8);
  if ( !v9 )
    return 63;
  v10 = *v9;
  result = 4;
  v11 = v10 & 3;
  if ( v11 )
  {
    result = 61;
    if ( (v11 & 2) != 0 )
      return 63;
  }
  return result;
}
