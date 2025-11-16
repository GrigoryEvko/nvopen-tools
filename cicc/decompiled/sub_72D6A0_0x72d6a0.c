// Function: sub_72D6A0
// Address: 0x72d6a0
//
__int64 __fastcall sub_72D6A0(_QWORD *a1)
{
  _QWORD *v1; // rcx
  __int64 *v2; // rax
  __int64 *v3; // rdx
  __int64 v4; // r13
  _QWORD *v6; // rax
  __int64 v7; // r13

  v1 = 0;
  v2 = (__int64 *)a1[15];
  if ( v2 )
  {
    if ( *((_BYTE *)v2 + 16) == 1 )
    {
LABEL_5:
      v4 = v2[1];
      if ( v1 )
      {
        *v1 = *v2;
        *v2 = a1[15];
        a1[15] = v2;
      }
      if ( v4 )
        return v4;
    }
    else
    {
      while ( 1 )
      {
        v3 = (__int64 *)*v2;
        v1 = v2;
        if ( !*v2 )
          break;
        v2 = (__int64 *)*v2;
        if ( *((_BYTE *)v3 + 16) == 1 )
          goto LABEL_5;
      }
    }
  }
  v6 = sub_7259C0(6);
  *((_BYTE *)v6 + 168) |= 3u;
  v7 = (__int64)v6;
  v6[20] = a1;
  sub_8D6090(v6);
  sub_728520(a1, 1, v7);
  return v7;
}
