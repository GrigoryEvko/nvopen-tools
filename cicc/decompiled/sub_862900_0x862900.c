// Function: sub_862900
// Address: 0x862900
//
__int64 __fastcall sub_862900(__int64 a1, int a2)
{
  char v3; // dl
  __int64 v4; // rbx
  char v5; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // edi
  __int64 i; // rax
  __int64 v11; // rax
  unsigned __int8 v12; // cl
  _BOOL4 v13; // ebx
  _QWORD *v14; // rax
  __int64 v15; // rax
  int v16; // r13d
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 j; // rax
  _QWORD *v24; // rax
  bool v25; // zf

  v3 = *(_BYTE *)(a1 + 89);
  if ( !*qword_4D03FD0 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 4) == 0
      || *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 152) + 168LL) + 40LL)
      || !(unsigned int)sub_8D23B0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL)) )
    {
      if ( *(_BYTE *)(a1 + 174) != 1
        || (*(_BYTE *)(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL) + 96LL) + 183LL) & 0x10) == 0
        || (v19 = *(__int64 **)(sub_72B840(a1) + 48)) == 0 )
      {
LABEL_17:
        v3 = *(_BYTE *)(a1 + 89);
        if ( (*(_BYTE *)(a1 + 207) & 0x40) != 0 )
          goto LABEL_2;
        if ( a2 )
        {
          if ( (v3 & 4) != 0 )
          {
            v21 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
            if ( *(_BYTE *)(v21 + 140) == 9 && (*(_BYTE *)(*(_QWORD *)(v21 + 168) + 109LL) & 0x20) != 0 )
              goto LABEL_2;
          }
        }
        if ( (v3 & 1) == 0 )
          goto LABEL_25;
        v9 = 0;
        for ( i = dword_4F04C64; ; i = *(int *)(v11 + 552) )
        {
          v11 = qword_4F04C68[0] + 776 * i;
          v12 = *(_BYTE *)(v11 + 4);
          if ( v12 > 4u )
          {
            if ( v12 == 6 )
            {
              v20 = *(_QWORD *)(v11 + 208);
              if ( (*(_BYTE *)(v20 + 141) & 0x20) != 0 )
              {
                v9 = 1;
              }
              else
              {
                if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v20 + 96LL) + 181LL) & 2) != 0
                  && !*(_QWORD *)(*(_QWORD *)(v20 + 168) + 240LL) )
                {
                  goto LABEL_2;
                }
                if ( (*(_BYTE *)(v20 + 89) & 1) == 0 )
                {
LABEL_24:
                  if ( v9 )
                    goto LABEL_2;
LABEL_25:
                  if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
                    goto LABEL_2;
                  v13 = 1;
                  v14 = (_QWORD *)sub_72B840(a1);
                  if ( !v14[13] && !v14[20] && !v14[14] )
                    v13 = (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u;
                  if ( (*(_BYTE *)(a1 + 89) & 2) != 0 )
                    v15 = sub_72F070(a1);
                  else
                    v15 = *(_QWORD *)(a1 + 40);
                  if ( !v15 )
                  {
LABEL_68:
                    if ( (*(_BYTE *)(a1 + 205) & 0x40) == 0 )
                    {
                      if ( (*(_BYTE *)(a1 + 195) & 8) != 0
                        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0
                        || dword_4F077C4 != 2
                        || sub_7217A0() )
                      {
                        return 0;
                      }
                      v22 = (_QWORD *)sub_72B840(a1);
                      if ( (!dword_4D04278 || *(_BYTE *)(a1 + 172) != 2 || !v22[14] && !v22[13] && !v22[20])
                        && (!dword_4D04440 || !(unsigned int)sub_85C070(v22)) )
                      {
                        if ( !dword_4D048B8 )
                          goto LABEL_102;
                        for ( j = *(_QWORD *)(a1 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                          ;
                        if ( !*(_QWORD *)(*(_QWORD *)(j + 168) + 56LL)
                          || !(unsigned int)sub_80CED0(*(_QWORD *)(a1 + 152)) )
                        {
LABEL_102:
                          if ( !(unsigned int)sub_736990(a1) && !(unsigned int)sub_8D96E0(*(_QWORD *)(a1 + 152)) )
                            return 0;
                        }
                      }
                      v24 = (_QWORD *)sub_823970(16);
                      *v24 = 0;
                      v25 = qword_4F5FCE8 == 0;
                      v24[1] = a1;
                      if ( v25 )
                        qword_4F5FCE8 = (__int64)v24;
                      else
                        *(_QWORD *)qword_4F5FCE0 = v24;
                      qword_4F5FCE0 = (__int64)v24;
                    }
                    goto LABEL_62;
                  }
                  v16 = 0;
                  while ( 2 )
                  {
                    if ( (unsigned __int8)(*(_BYTE *)(v15 + 28) - 6) > 1u )
                    {
                      v15 = *(_QWORD *)(v15 + 16);
                    }
                    else
                    {
                      v17 = *(_QWORD *)(v15 + 32);
                      if ( v16 )
                        goto LABEL_37;
                      if ( **(_QWORD **)(v17 + 168) )
                      {
                        v16 = 1;
                        goto LABEL_37;
                      }
                      if ( v13 )
                      {
LABEL_37:
                        if ( (*(_BYTE *)(v17 + 177) & 4) != 0
                          && !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v17 + 96LL) + 168LL) )
                        {
                          goto LABEL_62;
                        }
                      }
                      if ( *(_BYTE *)(v17 + 140) == 9 )
                      {
                        v18 = *(_QWORD *)(v17 + 168);
                        if ( (*(_DWORD *)(v18 + 108) & 0x82000) == 0x82000
                          && (*(_BYTE *)(*(_QWORD *)(v18 + 240) + 170LL) & 2) != 0 )
                        {
                          goto LABEL_62;
                        }
                      }
                      if ( (unsigned int)sub_8D8C50(v17, sub_85B9B0, sub_85B040, 24) )
                        goto LABEL_62;
                      if ( (*(_BYTE *)(v17 + 89) & 2) != 0 )
                        v15 = sub_72F070(v17);
                      else
                        v15 = *(_QWORD *)(v17 + 40);
                    }
                    if ( !v15 )
                      goto LABEL_68;
                    continue;
                  }
                }
              }
            }
          }
          else if ( (unsigned __int8)(v12 - 1) > 1u )
          {
            goto LABEL_24;
          }
        }
      }
      while ( (*((_BYTE *)v19 + 9) & 8) == 0 || v19[3] )
      {
        v19 = (__int64 *)*v19;
        if ( !v19 )
          goto LABEL_17;
      }
    }
LABEL_62:
    v3 = *(_BYTE *)(a1 + 89);
  }
LABEL_2:
  if ( (v3 & 2) != 0 )
    v4 = sub_72F070(a1);
  else
    v4 = *(_QWORD *)(a1 + 40);
  for ( ; v4; v4 = *(_QWORD *)(v4 + 16) )
  {
    while ( 1 )
    {
      v5 = *(_BYTE *)(v4 + 28);
      if ( v5 != 17 )
        break;
      *(_BYTE *)(*(_QWORD *)(v4 + 32) + 205LL) |= 0x40u;
      v4 = *(_QWORD *)(v4 + 16);
      if ( !v4 )
        goto LABEL_10;
    }
    if ( v5 == 6 )
    {
      v7 = *(_QWORD *)(v4 + 32);
      if ( (*(_BYTE *)(v7 + 89) & 1) != 0 )
      {
        v8 = sub_72B7D0(v7);
        if ( !v8 )
          break;
        *(_BYTE *)(v8 + 205) |= 0x40u;
      }
    }
  }
LABEL_10:
  unk_4F04C20 = 1;
  return 1;
}
