// Function: sub_2099010
// Address: 0x2099010
//
_QWORD *__fastcall sub_2099010(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rsi
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = *a2;
  v4 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *(_QWORD *)(v4 + 8);
  if ( v5 )
  {
    while ( 1 )
    {
      v6 = sub_1648700(v5);
      if ( *((_BYTE *)v6 + 16) != 78 )
        goto LABEL_3;
      v7 = *(v6 - 3);
      if ( *(_BYTE *)(v7 + 16) || (*(_BYTE *)(v7 + 33) & 0x20) == 0 || *(_DWORD *)(v7 + 36) != 76 )
        goto LABEL_3;
      v14[0] = v6;
      v8 = (_BYTE *)a1[1];
      if ( v8 == (_BYTE *)a1[2] )
      {
        sub_2098E80((__int64)a1, v8, v14);
LABEL_3:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
      }
      else
      {
        if ( v8 )
        {
          *(_QWORD *)v8 = v6;
          v8 = (_BYTE *)a1[1];
        }
        a1[1] = v8 + 8;
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
      }
    }
  }
  if ( v4 )
  {
    if ( (v3 & 4) == 0 )
    {
      v9 = *(_QWORD *)(sub_15F6E60(*a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
      if ( v9 )
      {
        while ( 1 )
        {
          v10 = sub_1648700(v9);
          if ( *((_BYTE *)v10 + 16) != 78 )
            goto LABEL_16;
          v11 = *(v10 - 3);
          if ( *(_BYTE *)(v11 + 16) || (*(_BYTE *)(v11 + 33) & 0x20) == 0 || *(_DWORD *)(v11 + 36) != 76 )
            goto LABEL_16;
          v14[0] = v10;
          v12 = (_BYTE *)a1[1];
          if ( v12 == (_BYTE *)a1[2] )
          {
            sub_2098E80((__int64)a1, v12, v14);
LABEL_16:
            v9 = *(_QWORD *)(v9 + 8);
            if ( !v9 )
              return a1;
          }
          else
          {
            if ( v12 )
            {
              *(_QWORD *)v12 = v10;
              v12 = (_BYTE *)a1[1];
            }
            a1[1] = v12 + 8;
            v9 = *(_QWORD *)(v9 + 8);
            if ( !v9 )
              return a1;
          }
        }
      }
    }
  }
  return a1;
}
