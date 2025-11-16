// Function: sub_1358E20
// Address: 0x1358e20
//
__int64 __fastcall sub_1358E20(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12

  v2 = 0;
  v3 = a1 + 8;
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 != a1 + 8 )
  {
    while ( 1 )
    {
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 8);
      if ( *(_QWORD *)(v5 + 32) || !(unsigned __int8)sub_1358B50(v5, a2, *(_QWORD **)a1) )
        goto LABEL_3;
      if ( !v2 )
        break;
      if ( *(_QWORD *)(v5 + 32) )
      {
LABEL_3:
        if ( v3 == v4 )
          return v2;
      }
      else
      {
        sub_1357740(v2, v5, a1);
        if ( v3 == v4 )
          return v2;
      }
    }
    v2 = v5;
    goto LABEL_3;
  }
  return v2;
}
