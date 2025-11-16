// Function: sub_830310
// Address: 0x830310
//
__int64 __fastcall sub_830310(_QWORD *a1, __int64 *a2, int a3, __int64 *a4)
{
  __int64 v7; // rax
  char v8; // r10
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  char v12; // di
  unsigned int v13; // r8d
  __int64 v14; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  char v21; // cl
  char v22; // al
  __int64 i; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // r10
  __int64 m; // r13
  __int64 v30; // rdx
  __int64 v31; // rdi
  int v32; // edx
  char v33; // di
  char v34; // dl
  __int64 v35; // rdx
  char v36; // di
  __int64 v37; // rdx
  char j; // dl
  __int64 v39; // rax
  __int64 k; // rdi
  __int64 v41; // rax
  __int64 v42; // rsi
  char v43; // dl
  __int64 v44; // rcx
  __int64 v45; // [rsp+8h] [rbp-28h]

  v7 = qword_4F04C68[0] + 776LL * (int)dword_4F04C60;
  v8 = *(_BYTE *)(v7 + 4);
  if ( v8 == 1 )
  {
    while ( 1 )
    {
      while ( *(_BYTE *)(v7 - 772) == 6 )
      {
        v10 = *(_QWORD *)(v7 - 568);
        if ( *(_BYTE *)(v10 + 140) != 9 || (*(_BYTE *)(*(_QWORD *)(v10 + 168) + 109LL) & 0x20) == 0 )
          break;
        v43 = *(_BYTE *)(v7 - 1548);
        v7 -= 1552;
        if ( v43 != 1 )
          goto LABEL_10;
      }
      v9 = *(_QWORD *)(v7 + 624);
      if ( !v9 || (*(_BYTE *)(v9 + 133) & 1) == 0 )
        break;
      v43 = *(_BYTE *)(v7 - 772);
      v7 -= 776;
      if ( v43 != 1 )
        goto LABEL_10;
    }
LABEL_23:
    v16 = *(int *)(v7 + 400);
    if ( (_DWORD)v16 == -1 )
      goto LABEL_12;
    v11 = *(_QWORD *)(qword_4F04C68[0] + 776 * v16 + 184);
    if ( !v11 )
      goto LABEL_12;
LABEL_25:
    v17 = *(_QWORD *)(v11 + 32);
    if ( a3 )
    {
      if ( (*(_BYTE *)(v17 + 206) & 2) != 0 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(v17 + 40) + 32LL);
        v19 = *(_QWORD *)(v18 + 48);
        if ( !v19 )
        {
LABEL_93:
          v42 = *(_QWORD *)(v18 + 168);
LABEL_91:
          if ( (*(_BYTE *)(v42 + 110) & 0x10) == 0 )
            goto LABEL_16;
          v7 = sub_8D71C0(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL), *(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL));
          v14 = 0;
          v13 = 1;
LABEL_17:
          if ( a1 )
            goto LABEL_18;
          goto LABEL_19;
        }
        while ( 1 )
        {
          v22 = *(_BYTE *)(v18 + 89) & 4;
          if ( (*(_BYTE *)(v19 + 206) & 2) == 0 )
          {
            if ( v22 )
            {
              v42 = *(_QWORD *)(v18 + 168);
              v20 = *(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL);
              v21 = *(_BYTE *)(v20 + 140);
              goto LABEL_90;
            }
            goto LABEL_35;
          }
          if ( v22 )
          {
            v20 = *(_QWORD *)(*(_QWORD *)(v18 + 40) + 32LL);
            v21 = *(_BYTE *)(v20 + 140);
            if ( v21 != 9 || (*(_BYTE *)(*(_QWORD *)(v20 + 168) + 109LL) & 0x20) == 0 )
              break;
          }
          v18 = *(_QWORD *)(*(_QWORD *)(v19 + 40) + 32LL);
          v19 = *(_QWORD *)(v18 + 48);
          if ( !v19 )
            goto LABEL_93;
        }
        v42 = *(_QWORD *)(v18 + 168);
LABEL_90:
        if ( v21 != 9 || (*(_BYTE *)(*(_QWORD *)(v20 + 168) + 109LL) & 0x20) == 0 )
          goto LABEL_91;
LABEL_35:
        for ( i = *(_QWORD *)(v19 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v7 = *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL);
        if ( !v7 )
        {
LABEL_86:
          v14 = 0;
          v13 = 0;
          goto LABEL_17;
        }
        v24 = sub_72B840(v19);
        v13 = 1;
        v14 = *(_QWORD *)(v24 + 64);
LABEL_39:
        if ( v14 )
          v7 = *(_QWORD *)(v14 + 120);
        else
LABEL_68:
          v7 = 0;
        goto LABEL_17;
      }
    }
    else if ( (*(_BYTE *)(v17 + 206) & 2) != 0 )
    {
      goto LABEL_16;
    }
    v14 = *(_QWORD *)(v11 + 64);
    v13 = v14 != 0;
    goto LABEL_39;
  }
  v43 = *(_BYTE *)(v7 + 4);
LABEL_10:
  if ( v43 != 17 )
  {
    v8 = v43;
    goto LABEL_23;
  }
  v11 = *(_QWORD *)(v7 + 184);
  v8 = 17;
  if ( v11 )
    goto LABEL_25;
LABEL_12:
  v12 = *(_BYTE *)(v7 + 11);
  v13 = dword_4D048B4;
  if ( (!(dword_4D048B4 | unk_4D0442C) || v8 != 1) && v12 >= 0 )
  {
    if ( (*(_BYTE *)(v7 + 12) & 0x10) != 0 )
    {
      for ( j = *(_BYTE *)(v7 + 4); j == 14; j = *(_BYTE *)(v7 + 4) )
      {
        v39 = *(int *)(v7 + 552);
        if ( (_DWORD)v39 == -1 )
          BUG();
        v7 = qword_4F04C68[0] + 776 * v39;
      }
      if ( j == 9 )
      {
        v7 = *(_QWORD *)(v7 + 216);
        if ( v7 )
        {
          v41 = *(_QWORD *)(v7 + 152);
          for ( k = v41; *(_BYTE *)(v41 + 140) == 12; v41 = *(_QWORD *)(v41 + 160) )
            ;
          v7 = *(_QWORD *)(*(_QWORD *)(v41 + 168) + 40LL);
          if ( v7 )
          {
            v7 = sub_8D71D0(k);
            v14 = 0;
            v13 = 1;
            goto LABEL_17;
          }
        }
        goto LABEL_86;
      }
    }
LABEL_16:
    v14 = 0;
    v13 = 0;
    v7 = 0;
    goto LABEL_17;
  }
  while ( 1 )
  {
    v27 = *(_BYTE *)(v7 + 4);
    if ( v27 != 1 )
    {
      if ( v12 >= 0 )
        goto LABEL_16;
      if ( (unsigned __int8)(v27 - 6) <= 1u )
      {
        v25 = *(_QWORD *)(v7 + 208);
        if ( *(_BYTE *)(v25 + 140) != 9 || (*(_BYTE *)(*(_QWORD *)(v25 + 168) + 109LL) & 0x20) == 0 )
        {
          v7 = sub_8D71C0(v25, *(_QWORD *)(v7 + 208));
          v14 = 0;
          v13 = 1;
          goto LABEL_17;
        }
      }
      goto LABEL_45;
    }
    if ( (v12 & 0x40) == 0 )
      goto LABEL_45;
    v28 = *(_QWORD *)(v7 + 624);
    m = *(_QWORD *)(v7 + 208);
    v30 = *(_QWORD *)v28;
    if ( *(_QWORD *)v28 && (*(_BYTE *)(v28 + 121) & 0x40) == 0 )
    {
      v36 = *(_BYTE *)(v30 + 80);
      if ( (unsigned __int8)(v36 - 10) <= 1u )
      {
        v37 = *(_QWORD *)(v30 + 88);
      }
      else
      {
        if ( v36 != 20 )
          goto LABEL_52;
        v37 = *(_QWORD *)(*(_QWORD *)(v30 + 88) + 176LL);
      }
      for ( m = *(_QWORD *)(v37 + 152); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
    }
LABEL_52:
    if ( (*(_BYTE *)(v28 + 123) & 0x10) != 0 )
    {
      if ( !unk_4D0442C )
      {
        v13 = 0;
        v14 = 0;
        v7 = 0;
        goto LABEL_17;
      }
    }
    else if ( !dword_4D048B4 )
    {
      v14 = 0;
      goto LABEL_68;
    }
    v31 = *(_QWORD *)(m + 168);
    v32 = *(_BYTE *)(v31 + 21) & 1;
    if ( (*(_BYTE *)(v31 + 21) & 1) != 0 )
      break;
    v33 = *(_BYTE *)(v28 + 129);
    if ( (v33 & 0x40) == 0 )
      goto LABEL_57;
LABEL_45:
    v26 = *(int *)(v7 + 552);
    if ( (_DWORD)v26 == -1 )
      goto LABEL_16;
    v7 = qword_4F04C68[0] + 776 * v26;
    if ( !v7 )
      goto LABEL_86;
    v12 = *(_BYTE *)(v7 + 11);
  }
  if ( (*(_BYTE *)(v31 + 17) & 4) != 0 )
    goto LABEL_45;
  v33 = *(_BYTE *)(v28 + 129);
  if ( (v33 & 0x40) != 0 )
    goto LABEL_95;
LABEL_57:
  if ( v33 >= 0 )
    goto LABEL_45;
  if ( v32 )
  {
LABEL_95:
    if ( a2 )
    {
      v7 = sub_8D71D0(m);
      goto LABEL_97;
    }
  }
  else
  {
    if ( a4 )
    {
      v34 = *(_BYTE *)(v28 + 130);
      if ( (v34 & 2) == 0 )
      {
        v44 = *a4;
        v45 = v7;
        *(_BYTE *)(v28 + 130) = v34 | 2;
        *(_QWORD *)(v28 + 440) = v44;
        sub_643E40((__int64)sub_82BAE0, v28, 0);
        v7 = v45;
      }
    }
    if ( a2 )
    {
      v35 = v7 - 1552;
      if ( *(_BYTE *)(v7 - 772) != 9 )
        v35 = v7 - 776;
      *(_QWORD *)(*(_QWORD *)(m + 168) + 40LL) = *(_QWORD *)(v35 + 208);
      *(_BYTE *)(*(_QWORD *)(m + 168) + 21LL) |= 1u;
      v7 = sub_8D71D0(m);
      *(_QWORD *)(*(_QWORD *)(m + 168) + 40LL) = 0;
      *(_BYTE *)(*(_QWORD *)(m + 168) + 21LL) &= ~1u;
LABEL_97:
      v14 = 0;
      v13 = 1;
      if ( a1 )
        goto LABEL_18;
      goto LABEL_20;
    }
  }
  v13 = 1;
  if ( !a1 )
    return v13;
  v14 = 0;
  v13 = 1;
  v7 = 0;
LABEL_18:
  *a1 = v14;
LABEL_19:
  if ( a2 )
LABEL_20:
    *a2 = v7;
  return v13;
}
