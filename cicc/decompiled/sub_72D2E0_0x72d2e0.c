// Function: sub_72D2E0
// Address: 0x72d2e0
//
__int64 __fastcall sub_72D2E0(_QWORD *a1)
{
  __int64 *v1; // rax
  _QWORD *v2; // rcx
  __int64 *v3; // rdx
  __int64 v4; // r13
  _QWORD *v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // rax
  _QWORD *v9; // r13

  if ( a1 )
  {
    v1 = (__int64 *)a1[15];
    v2 = 0;
    if ( v1 )
    {
      if ( *((_BYTE *)v1 + 16) == 5 )
      {
LABEL_6:
        v4 = v1[1];
        if ( v2 )
        {
          *v2 = *v1;
          *v1 = a1[15];
          a1[15] = v1;
        }
        if ( v4 )
          return v4;
      }
      else
      {
        while ( 1 )
        {
          v3 = (__int64 *)*v1;
          v2 = v1;
          if ( !*v1 )
            break;
          v1 = (__int64 *)*v1;
          if ( *((_BYTE *)v3 + 16) == 5 )
            goto LABEL_6;
        }
      }
    }
    v6 = sub_7259C0(6);
    v6[20] = a1;
    v7 = (__int64)v6;
    sub_8D6090(v6);
    sub_728520(a1, 5, v7);
    return v7;
  }
  else
  {
    v8 = sub_7259C0(6);
    v8[20] = 0;
    v9 = v8;
    sub_8D6090(v8);
    return (__int64)v9;
  }
}
