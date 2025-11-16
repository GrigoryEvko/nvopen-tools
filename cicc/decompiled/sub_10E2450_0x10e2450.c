// Function: sub_10E2450
// Address: 0x10e2450
//
_QWORD *__fastcall sub_10E2450(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( *(_BYTE *)v4 != 85 )
        goto LABEL_3;
      v5 = *(_QWORD *)(v4 - 32);
      if ( !v5
        || *(_BYTE *)v5
        || *(_QWORD *)(v5 + 24) != *(_QWORD *)(v4 + 80)
        || (*(_BYTE *)(v5 + 33) & 0x20) == 0
        || *(_DWORD *)(v5 + 36) != 149 )
      {
        goto LABEL_3;
      }
      v12[0] = *(_QWORD *)(v3 + 24);
      v6 = (_BYTE *)a1[1];
      if ( v6 == (_BYTE *)a1[2] )
      {
        sub_10E22C0((__int64)a1, v6, v12);
LABEL_3:
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          break;
      }
      else
      {
        if ( v6 )
        {
          *(_QWORD *)v6 = v4;
          v6 = (_BYTE *)a1[1];
        }
        a1[1] = v6 + 8;
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          break;
      }
    }
  }
  if ( *(_BYTE *)a2 == 34 )
  {
    v7 = *(_QWORD *)(sub_B4B0F0(a2) + 16);
    if ( v7 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 + 24);
        if ( *(_BYTE *)v8 != 85 )
          goto LABEL_17;
        v9 = *(_QWORD *)(v8 - 32);
        if ( !v9
          || *(_BYTE *)v9
          || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v8 + 80)
          || (*(_BYTE *)(v9 + 33) & 0x20) == 0
          || *(_DWORD *)(v9 + 36) != 149 )
        {
          goto LABEL_17;
        }
        v12[0] = *(_QWORD *)(v7 + 24);
        v10 = (_BYTE *)a1[1];
        if ( v10 == (_BYTE *)a1[2] )
        {
          sub_10E22C0((__int64)a1, v10, v12);
LABEL_17:
          v7 = *(_QWORD *)(v7 + 8);
          if ( !v7 )
            return a1;
        }
        else
        {
          if ( v10 )
          {
            *(_QWORD *)v10 = v8;
            v10 = (_BYTE *)a1[1];
          }
          a1[1] = v10 + 8;
          v7 = *(_QWORD *)(v7 + 8);
          if ( !v7 )
            return a1;
        }
      }
    }
  }
  return a1;
}
