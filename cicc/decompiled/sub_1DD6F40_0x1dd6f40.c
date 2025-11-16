// Function: sub_1DD6F40
// Address: 0x1dd6f40
//
__int64 *__fastcall sub_1DD6F40(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  __int16 v5; // ax
  __int64 v6; // rsi
  __int64 i; // rbx
  __int16 v8; // ax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // rsi
  _QWORD v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2 + 24;
  *a1 = 0;
  v3 = sub_1DD5EE0(a2);
  if ( v3 != a2 + 24 )
  {
    v4 = v3;
    while ( 1 )
    {
      v5 = *(_WORD *)(v4 + 46);
      if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
      {
        if ( (unsigned __int8)sub_1E15D00(v4, 128, 1) )
          break;
        goto LABEL_5;
      }
      if ( (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL) & 0x80u) != 0LL )
        break;
LABEL_5:
      if ( (*(_BYTE *)v4 & 4) != 0 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v2 )
          return a1;
      }
      else
      {
        while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
          v4 = *(_QWORD *)(v4 + 8);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v2 )
          return a1;
      }
    }
    if ( a1 != (__int64 *)(v4 + 64) )
    {
      if ( *a1 )
        sub_161E7C0((__int64)a1, *a1);
      v6 = *(_QWORD *)(v4 + 64);
      *a1 = v6;
      if ( v6 )
        sub_1623A60((__int64)a1, v6, 2);
    }
    if ( (*(_BYTE *)v4 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
        v4 = *(_QWORD *)(v4 + 8);
    }
    for ( i = *(_QWORD *)(v4 + 8); i != v2; i = *(_QWORD *)(i + 8) )
    {
      v8 = *(_WORD *)(i + 46);
      if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(i + 16) + 8LL) & 0x80u) != 0LL )
        {
LABEL_23:
          v9 = sub_15C70A0(i + 64);
          v10 = sub_15C70A0((__int64)a1);
          v11 = sub_15BA070(v10, v9, 0);
          sub_15C7080(v14, v11);
          if ( *a1 )
            sub_161E7C0((__int64)a1, *a1);
          v12 = (unsigned __int8 *)v14[0];
          *a1 = v14[0];
          if ( v12 )
            sub_1623210((__int64)v14, v12, (__int64)a1);
        }
      }
      else if ( (unsigned __int8)sub_1E15D00(i, 128, 1) )
      {
        goto LABEL_23;
      }
      if ( (*(_BYTE *)i & 4) == 0 )
      {
        while ( (*(_BYTE *)(i + 46) & 8) != 0 )
          i = *(_QWORD *)(i + 8);
      }
    }
  }
  return a1;
}
