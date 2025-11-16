// Function: sub_FD7B50
// Address: 0xfd7b50
//
__int64 __fastcall sub_FD7B50(__int64 **a1, unsigned __int8 *a2)
{
  __int64 v2; // r15
  __int64 **v3; // r13
  __int64 *v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r9

  v2 = 0;
  v3 = a1 + 1;
  v4 = a1[2];
  if ( v4 != (__int64 *)(a1 + 1) )
  {
    while ( 1 )
    {
      v5 = (__int64)v4;
      v4 = (__int64 *)v4[1];
      if ( *(_QWORD *)(v5 + 16) || !(unsigned __int8)sub_FD60A0(v5, a2, (_QWORD **)*a1) )
        goto LABEL_3;
      if ( v2 )
      {
        sub_FD7340(v2, v5, (__int64)a1, *a1, v6, v7);
        if ( v3 == (__int64 **)v4 )
          return v2;
      }
      else
      {
        v2 = v5;
LABEL_3:
        if ( v3 == (__int64 **)v4 )
          return v2;
      }
    }
  }
  return v2;
}
