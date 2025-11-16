// Function: sub_5E9940
// Address: 0x5e9940
//
__int64 __fastcall sub_5E9940(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD *v7; // r8
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r14
  _QWORD *v11; // r8
  _QWORD *v12; // r15
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  bool v20; // cf
  unsigned __int64 v21; // rdx
  _QWORD *v22; // r8
  __int64 v23; // rax
  _QWORD *v24; // rsi
  _QWORD *v25; // rax
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  bool v31; // [rsp+7h] [rbp-49h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  _QWORD *v33; // [rsp+18h] [rbp-38h]
  _QWORD *v34; // [rsp+18h] [rbp-38h]
  _QWORD *v35; // [rsp+18h] [rbp-38h]

  v2 = *a1;
  v3 = a1[57];
  if ( !*a1 || (unsigned __int8)(*(_BYTE *)(v2 + 80) - 10) > 1u )
    goto LABEL_3;
  v7 = *(_QWORD **)(v3 + 128);
  v8 = *(_QWORD *)(v3 + 24);
  v32 = *(_QWORD *)(v2 + 88);
  v31 = (*(_BYTE *)(v2 + 83) & 0x40) != 0;
  if ( (*(_BYTE *)(v2 + 81) & 0x10) != 0 )
  {
    v35 = *(_QWORD **)(v3 + 128);
    sub_866000(*(_QWORD *)(v2 + 64), (*((_BYTE *)a1 + 130) & 8) != 0, 1);
    v7 = v35;
  }
  else if ( (*((_BYTE *)a1 + 127) & 0x10) != 0 )
  {
    *(_BYTE *)(v2 + 83) |= 0x40u;
  }
  v33 = v7;
  sub_8600D0(14, 0xFFFFFFFFLL, 0, v32);
  v9 = sub_877FE0(*a1);
  sub_8600D0(1, *(unsigned int *)(v3 + 64), v9, 0);
  *(_QWORD *)(unk_4F04C68 + 776LL * dword_4F04C64 + 624) = a1;
  if ( v8 )
  {
    v10 = 0;
    do
    {
      while ( *(_BYTE *)(v8 + 80) != 18 || *(_DWORD *)(*(_QWORD *)(v8 + 88) + 120LL) <= *(_DWORD *)(v33[6] + 36LL) )
      {
        v8 = *(_QWORD *)(v8 + 16);
        if ( !v8 )
          goto LABEL_17;
      }
      *(_BYTE *)(v8 + 83) |= 0x40u;
      if ( !v10 )
        v10 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v8 );
LABEL_17:
    sub_886000(*(_QWORD *)(v3 + 24));
    v11 = v33;
    if ( !v33 )
      goto LABEL_51;
    while ( 1 )
    {
LABEL_18:
      v12 = (_QWORD *)v11[6];
      v13 = v12;
      if ( !a1[37] )
        goto LABEL_28;
      v14 = *(_QWORD **)(*(_QWORD *)(v32 + 152) + 168LL);
      v12 = (_QWORD *)*v14;
      if ( v13 )
      {
        v15 = 0;
        do
        {
          v13 = (_QWORD *)*v13;
          ++v15;
        }
        while ( v13 );
        if ( !v12 )
        {
          v17 = 0;
          goto LABEL_25;
        }
      }
      else
      {
        if ( !v12 )
          goto LABEL_28;
        v15 = 0;
      }
      v16 = (_QWORD *)*v14;
      v17 = 0;
      do
      {
        v16 = (_QWORD *)*v16;
        ++v17;
      }
      while ( v16 );
LABEL_25:
      if ( v15 > v17 )
      {
        v12 = 0;
      }
      else
      {
        v18 = v17 - v15;
        v19 = v18 - 1;
        if ( v18 )
        {
          do
          {
            v12 = (_QWORD *)*v12;
            v20 = v19-- == 0;
          }
          while ( !v20 );
        }
      }
LABEL_28:
      v34 = v11;
      sub_7BC000(v11 + 1);
      sub_6794F0(v12, *a1, 1);
      v22 = v34;
      if ( v10 )
      {
        while ( *(_BYTE *)(v10 + 80) != 18 )
        {
          v10 = *(_QWORD *)(v10 + 16);
          if ( !v10 )
            goto LABEL_33;
        }
        *(_BYTE *)(v10 + 83) &= ~0x40u;
      }
LABEL_33:
      if ( unk_4F04C3C )
        goto LABEL_46;
      v23 = *(_QWORD *)(v3 + 104);
      if ( *(_QWORD *)(v32 + 152) == v23 || !v23 )
        goto LABEL_46;
      v24 = **(_QWORD ***)(v23 + 168);
      if ( v12 )
      {
        v25 = v12;
        v26 = 0;
        do
        {
          v25 = (_QWORD *)*v25;
          ++v26;
        }
        while ( v25 );
        if ( v24 )
        {
LABEL_40:
          v27 = v24;
          v28 = 0;
          do
          {
            v27 = (_QWORD *)*v27;
            ++v28;
          }
          while ( v27 );
        }
        else
        {
          v28 = 0;
        }
        if ( v26 > v28 )
        {
          MEMORY[0x28] = sub_73BB50(v12[5], v24, v28, v32, v34);
          BUG();
        }
        v21 = v28 - v26;
        v29 = v21 - 1;
        if ( v21 )
        {
          do
          {
            v24 = (_QWORD *)*v24;
            v20 = v29-- == 0;
          }
          while ( !v20 );
        }
        goto LABEL_45;
      }
      if ( v24 )
      {
        v26 = 0;
        goto LABEL_40;
      }
LABEL_45:
      v30 = sub_73BB50(v12[5], v24, v21, v32, v34);
      v22 = v34;
      v24[5] = v30;
LABEL_46:
      v11 = (_QWORD *)*v22;
      if ( !v11 )
      {
        while ( v10 )
        {
          if ( *(_BYTE *)(v10 + 80) == 18 )
            *(_BYTE *)(v10 + 83) &= ~0x40u;
          v10 = *(_QWORD *)(v10 + 16);
LABEL_51:
          ;
        }
        goto LABEL_52;
      }
    }
  }
  v10 = 0;
  sub_886000(*(_QWORD *)(v3 + 24));
  v11 = v33;
  if ( v33 )
    goto LABEL_18;
LABEL_52:
  sub_863FC0();
  sub_863FC0();
  if ( (*(_BYTE *)(v2 + 81) & 0x10) != 0 )
  {
    sub_866010();
  }
  else if ( (*((_BYTE *)a1 + 127) & 0x10) != 0 )
  {
    *(_BYTE *)(v2 + 83) = *(_BYTE *)(v2 + 83) & 0xBF | (v31 << 6);
  }
LABEL_3:
  if ( (*(_BYTE *)(v3 + 184) & 2) == 0 )
    sub_679050(*(_QWORD *)(v3 + 128));
  v4 = qword_4CF8000;
  *(_QWORD *)(v3 + 128) = 0;
  *(_QWORD *)v3 = v4;
  v5 = a1[36];
  a1[57] = 0;
  qword_4CF8000 = v3;
  return sub_85E870(v5, v2);
}
