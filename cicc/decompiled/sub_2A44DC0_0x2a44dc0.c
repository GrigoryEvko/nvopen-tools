// Function: sub_2A44DC0
// Address: 0x2a44dc0
//
void __fastcall sub_2A44DC0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // r10
  _BYTE *v4; // r9
  int v5; // eax
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // rax
  __int64 v12; // r9

  if ( a2 == a3 )
    return;
  v3 = *(_BYTE **)(a2 + 16);
  v4 = *(_BYTE **)(a3 + 16);
  if ( *(_DWORD *)a2 != *(_DWORD *)a3 )
    return;
  v5 = *(_DWORD *)(a2 + 8);
  if ( v5 != 2 )
  {
    if ( v5 != 1 || *(_DWORD *)(a3 + 8) != 1 )
      return;
    if ( !v3 )
    {
      if ( *(_QWORD *)(a2 + 24)
        || (v9 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 56LL), v10 = *(_QWORD *)(v9 + 32),
                                                           v10 == *(_QWORD *)(v9 + 40) + 48LL)
        || !v10 )
      {
        v3 = 0;
        if ( v4 )
          goto LABEL_44;
LABEL_25:
        if ( *(_QWORD *)(a3 + 24)
          || (v11 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 56LL),
              v12 = *(_QWORD *)(v11 + 32),
              v12 == *(_QWORD *)(v11 + 40) + 48LL)
          || !v12 )
        {
          if ( v3 )
          {
            v4 = 0;
            if ( *v3 == 22 )
              goto LABEL_47;
            goto LABEL_28;
          }
        }
        else
        {
          v4 = (_BYTE *)(v12 - 24);
          if ( v3 )
          {
LABEL_12:
            if ( *v3 == 22 )
            {
              if ( v4 && *v4 != 22 )
                v4 = 0;
              goto LABEL_47;
            }
            if ( v4 )
            {
LABEL_33:
              if ( *v4 == 22 )
              {
                v3 = 0;
              }
              else if ( !v3 )
              {
                v3 = *(_BYTE **)(*(_QWORD *)(a2 + 24) + 24LL);
              }
              goto LABEL_47;
            }
            goto LABEL_28;
          }
          if ( v4 )
          {
LABEL_44:
            v3 = 0;
            goto LABEL_33;
          }
        }
        v3 = *(_BYTE **)(*(_QWORD *)(a2 + 24) + 24LL);
LABEL_28:
        v4 = *(_BYTE **)(*(_QWORD *)(a3 + 24) + 24LL);
LABEL_47:
        if ( v3 && *v3 == 22 )
        {
          if ( v4 )
          {
            if ( *v4 != 22 )
              nullsub_2022();
          }
        }
        else if ( !v4 || *v4 != 22 )
        {
          sub_B445A0((__int64)v3, (__int64)v4);
        }
        return;
      }
      v3 = (_BYTE *)(v10 - 24);
    }
    if ( v4 )
      goto LABEL_12;
    goto LABEL_25;
  }
  if ( *(_DWORD *)(a3 + 8) == 2 )
  {
    if ( v4 || (v8 = *(_QWORD *)(a3 + 24)) == 0 )
      v6 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 64LL);
    else
      v6 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 40LL);
    v7 = v6 ? *(_DWORD *)(v6 + 44) + 1 : 0;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 32LL) <= v7 )
      BUG();
  }
}
