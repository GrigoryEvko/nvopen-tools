// Function: sub_885620
// Address: 0x885620
//
void __fastcall sub_885620(__int64 a1, int a2, _DWORD *a3)
{
  char v4; // al
  __int64 v5; // rsi
  __int64 *v6; // rbx
  char v7; // r13
  char v8; // cl
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rcx
  char v18; // di
  _DWORD *v19; // rsi
  unsigned int v20; // edi
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rsi

  v4 = *(_BYTE *)(a1 + 81);
  if ( a2 != -1 )
  {
    v5 = qword_4F04C68[0] + 776LL * a2;
    v6 = *(__int64 **)(v5 + 24);
    v7 = *(_BYTE *)(v5 + 4);
    if ( !v6 )
      v6 = (__int64 *)(v5 + 32);
    *(_DWORD *)(a1 + 40) = *(_DWORD *)v5;
    v8 = v4 & 0x20;
    if ( dword_4F077C4 != 2 )
    {
LABEL_5:
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      if ( v8 )
        return;
      goto LABEL_10;
    }
    if ( v8 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      return;
    }
    if ( *(_BYTE *)(v5 + 4) == 6 )
    {
      v16 = **(__int64 ***)(v5 + 208);
      v17 = *v16;
      if ( *v16 == *(_QWORD *)a1 )
      {
        v18 = *(_BYTE *)(a1 + 80);
        if ( v18 == 8 )
        {
          if ( *(_QWORD *)(v16[12] + 8)
            || (v21 = *(_QWORD *)(a1 + 88)) != 0
            && ((v4 & 0x10) == 0
             || (v22 = *(_QWORD *)(a1 + 64), v23 = *(_QWORD *)(*(_QWORD *)(v21 + 40) + 32LL), v22 != v23)
             && (!v22 || !v23 || !dword_4F07588 || (v24 = *(_QWORD *)(v22 + 32), *(_QWORD *)(v23 + 32) != v24) || !v24)) )
          {
            if ( v17 != qword_4F600E0 )
              goto LABEL_46;
          }
        }
        else if ( v17 != qword_4F600E0 )
        {
          if ( v18 == 3 )
          {
            if ( !*(_BYTE *)(a1 + 104) )
              goto LABEL_46;
          }
          else
          {
            if ( v18 != 16 )
            {
              v19 = (_DWORD *)(a1 + 48);
              if ( (unsigned __int8)(v18 - 10) <= 1u || v18 == 17 )
              {
                v20 = 405;
                goto LABEL_48;
              }
LABEL_47:
              v20 = 280;
LABEL_48:
              sub_6851C0(v20, v19);
              *(_BYTE *)(a1 + 81) |= 0x20u;
              *a3 = 1;
              v8 = *(_BYTE *)(a1 + 81) & 0x20;
              goto LABEL_5;
            }
            if ( (*(_BYTE *)(a1 + 96) & 4) != 0 )
            {
LABEL_46:
              v19 = (_DWORD *)(a1 + 48);
              goto LABEL_47;
            }
          }
        }
      }
    }
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
LABEL_10:
    if ( *v6 )
    {
      *(_QWORD *)(v6[2] + 16) = a1;
      *(_QWORD *)(a1 + 24) = v6[2];
    }
    else
    {
      *v6 = a1;
    }
    v6[2] = a1;
    if ( (unsigned int)sub_8770E0(v7) )
    {
      v11 = (unsigned __int8 *)v6[17];
      if ( !v11 )
      {
        switch ( v7 )
        {
          case 0:
          case 3:
          case 4:
            v14 = 100;
            break;
          case 1:
          case 2:
          case 8:
          case 9:
            v14 = 10;
            break;
          case 6:
            v14 = 30;
            break;
          case 7:
          case 13:
          case 16:
            v14 = 5;
            break;
          case 15:
            v14 = 2;
            break;
          case 17:
            v14 = 20;
            break;
          default:
            sub_721090();
        }
        v15 = sub_881A70(0, v14, 14, 15, v9, v10);
        v6[17] = v15;
        v11 = (unsigned __int8 *)v15;
      }
      sub_885590((__int64 *)a1, v11);
    }
    return;
  }
  v12 = *(_QWORD *)(a1 + 64);
  if ( (v4 & 0x10) != 0 )
  {
    while ( *(_BYTE *)(v12 + 140) == 12 )
      v12 = *(_QWORD *)(v12 + 160);
    v7 = 6;
    v8 = v4 & 0x20;
    v6 = (__int64 *)(*(_QWORD *)(*(_QWORD *)v12 + 96LL) + 192LL);
    goto LABEL_5;
  }
  if ( !v12 )
  {
    *(_DWORD *)(a1 + 40) = -1;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    if ( (v4 & 0x20) != 0 )
      return;
    goto LABEL_21;
  }
  if ( (*(_BYTE *)(v12 + 124) & 1) != 0 )
  {
    v12 = sub_735B70(v12);
    v4 = *(_BYTE *)(a1 + 81);
  }
  *(_DWORD *)(a1 + 40) = *(_DWORD *)(*(_QWORD *)(v12 + 128) + 24LL);
  v6 = *(__int64 **)(*(_QWORD *)v12 + 96LL);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( (v4 & 0x20) == 0 )
  {
    if ( v6 )
    {
      v7 = 3;
      goto LABEL_10;
    }
LABEL_21:
    if ( qword_4D04970 )
    {
      v13 = qword_4F600B8;
      *(_QWORD *)(qword_4F600B8 + 16) = a1;
      *(_QWORD *)(a1 + 24) = v13;
    }
    else
    {
      qword_4D04970 = a1;
    }
    qword_4F600B8 = a1;
  }
}
