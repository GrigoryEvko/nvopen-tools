// Function: sub_1897310
// Address: 0x1897310
//
__int64 __fastcall sub_1897310(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // r12d
  unsigned __int8 v6; // al
  __int64 v7; // r15
  __int64 v9; // rdx

  v1 = a1 + 40;
  v2 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL));
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != a1 + 40 )
  {
    v4 = v2;
    v5 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v3 )
          BUG();
        v6 = *(_BYTE *)(v3 - 8);
        v7 = v3 - 24;
        if ( v6 == 78 )
        {
          v9 = *(_QWORD *)(v3 - 48);
          if ( *(_BYTE *)(v9 + 16) || (*(_BYTE *)(v9 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v9 + 36) - 35) > 3 )
            goto LABEL_20;
          goto LABEL_7;
        }
        if ( v6 != 56 )
          break;
        if ( !(unsigned __int8)sub_15FA1F0(v3 - 24) )
        {
          v6 = *(_BYTE *)(v3 - 8);
          if ( v6 == 78 )
          {
            v9 = *(_QWORD *)(v3 - 48);
LABEL_20:
            if ( *(_BYTE *)(v9 + 16)
              || (*(_BYTE *)(v9 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v9 + 36) - 116) > 1 )
            {
              v5 += sub_38504C0(v7 | 4, v4);
            }
            goto LABEL_7;
          }
LABEL_5:
          if ( v6 == 29 )
          {
            v5 += sub_38504C0(v7 & 0xFFFFFFFFFFFFFFFBLL, v4);
          }
          else if ( v6 == 27 )
          {
            v5 += 5 * ((*(_DWORD *)(v3 - 4) & 0xFFFFFFFu) >> 1);
          }
          else
          {
            v5 += 5;
          }
        }
LABEL_7:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v1 == v3 )
          return v5;
      }
      if ( (unsigned int)v6 - 24 > 0x20 )
      {
        if ( (unsigned int)v6 - 69 > 2 )
          goto LABEL_5;
        goto LABEL_7;
      }
      if ( v6 != 53 )
        goto LABEL_5;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v1 == v3 )
        return v5;
    }
  }
  return 0;
}
