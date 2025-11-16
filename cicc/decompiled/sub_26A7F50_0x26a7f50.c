// Function: sub_26A7F50
// Address: 0x26a7f50
//
__int64 __fastcall sub_26A7F50(__int64 a1, __int64 a2)
{
  int v4; // eax
  const char *v5; // rdx
  size_t v6; // rcx
  unsigned int v7; // eax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 *v14; // r11
  __int64 *v15; // r13
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // dl
  char v20; // r14
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int8 v24; // al
  __int64 v25; // rax
  unsigned int v26; // esi
  __int64 v27; // rdx
  __int64 *v28; // r13
  __int64 *v29; // r14
  __int64 v30; // rax
  __int64 v31; // rax
  bool v32; // zf
  int v33; // r14d
  unsigned __int64 v34; // rdi
  _QWORD *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+0h] [rbp-70h]
  char v47; // [rsp+Bh] [rbp-65h]
  __int64 v48; // [rsp+Ch] [rbp-64h]
  char v49; // [rsp+Ch] [rbp-64h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+14h] [rbp-5Ch]
  int v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+18h] [rbp-58h]
  unsigned int v54; // [rsp+2Ch] [rbp-44h] BYREF
  unsigned __int64 v55; // [rsp+30h] [rbp-40h]
  __int64 v56; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 120);
  v54 = 1;
  if ( v4 != 185 )
  {
    if ( v4 <= 185 )
    {
      if ( v4 == 14 )
      {
        v5 = "omp_target_num_teams";
        v6 = 20;
LABEL_5:
        v7 = sub_26A7D30(a1, a2, v5, v6);
        return sub_250C0B0(v54, v7);
      }
      if ( v4 == 15 )
      {
        v5 = "omp_target_thread_limit";
        v6 = 23;
        goto LABEL_5;
      }
LABEL_88:
      BUG();
    }
    if ( v4 != 186 )
      goto LABEL_88;
    v46 = *(_QWORD *)(a1 + 104);
    v47 = *(_BYTE *)(a1 + 112);
    v9 = *(_QWORD *)(a1 + 72);
    v10 = v9 & 3;
    v11 = v9 & 0xFFFFFFFFFFFFFFFCLL;
    if ( v10 == 3 )
      v11 = *(_QWORD *)(v11 + 24);
    v12 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 )
    {
      if ( v12 == 22 )
      {
        v11 = *(_QWORD *)(v11 + 24);
      }
      else if ( v12 <= 0x1Cu )
      {
        v11 = 0;
      }
      else
      {
        v11 = sub_B43CB0(v11);
      }
    }
    v56 = 0;
    v55 = v11 & 0xFFFFFFFFFFFFFFFCLL;
    nullsub_1518();
    v13 = sub_26A73D0(a2, v11 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 0, 1);
    if ( v13 && *(_BYTE *)(v13 + 337) )
    {
      v14 = *(__int64 **)(v13 + 376);
      v15 = &v14[*(unsigned int *)(v13 + 384)];
      if ( v14 == v15 )
      {
        v34 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
        if ( (*(_QWORD *)(a1 + 72) & 3LL) != 3 )
        {
          sub_BD5C60(v34);
LABEL_65:
          if ( *(_BYTE *)(a1 + 112) == v47 )
          {
            if ( !v47 )
            {
              v26 = 1;
              goto LABEL_34;
            }
LABEL_74:
            v26 = *(_QWORD *)(a1 + 104) == v46;
            goto LABEL_34;
          }
          goto LABEL_33;
        }
        v33 = 0;
        v48 = 0;
      }
      else
      {
        v16 = *(__int64 **)(v13 + 376);
        v48 = 0;
        v52 = 0;
        v51 = 0;
        do
        {
          v17 = *v16;
          v56 = 0;
          v55 = v17 & 0xFFFFFFFFFFFFFFFCLL;
          nullsub_1518();
          v18 = sub_26A73D0(a2, v55, 0, a1, 0, 1);
          if ( !v18 )
          {
            if ( !*(_BYTE *)(a1 + 112) )
              *(_BYTE *)(a1 + 112) = 1;
            goto LABEL_31;
          }
          v19 = *(_BYTE *)(v18 + 240);
          if ( *(_BYTE *)(v18 + 241) )
          {
            if ( v19 )
              ++v51;
            else
              ++v52;
          }
          else if ( v19 )
          {
            ++HIDWORD(v48);
          }
          else
          {
            LODWORD(v48) = v48 + 1;
          }
          ++v16;
        }
        while ( v15 != v16 );
        if ( v51 + v52 )
        {
          if ( (_DWORD)v48 + HIDWORD(v48) )
            goto LABEL_49;
          v33 = v51 | v52;
        }
        else
        {
          v33 = v51 | v52;
        }
        v34 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
        if ( (*(_QWORD *)(a1 + 72) & 3LL) != 3 )
        {
LABEL_63:
          v35 = (_QWORD *)sub_BD5C60(v34);
          if ( v33 )
          {
            v36 = sub_BCB2B0(v35);
            v37 = sub_ACD640(v36, 1, 0);
            v32 = *(_BYTE *)(a1 + 112) == 0;
            *(_QWORD *)(a1 + 104) = v37;
            if ( !v32 )
              goto LABEL_73;
          }
          else
          {
            if ( !v48 )
              goto LABEL_65;
            v42 = sub_BCB2B0(v35);
            v43 = sub_ACD640(v42, 0, 0);
            v32 = *(_BYTE *)(a1 + 112) == 0;
            *(_QWORD *)(a1 + 104) = v43;
            if ( !v32 )
              goto LABEL_73;
          }
          *(_BYTE *)(a1 + 112) = 1;
LABEL_73:
          if ( v47 )
            goto LABEL_74;
          goto LABEL_33;
        }
      }
      v34 = *(_QWORD *)(v34 + 24);
      goto LABEL_63;
    }
    goto LABEL_30;
  }
  v20 = *(_BYTE *)(a1 + 112);
  v53 = *(_QWORD *)(a1 + 104);
  v21 = *(_QWORD *)(a1 + 72);
  v22 = v21 & 3;
  v23 = v21 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v22 == 3 )
    v23 = *(_QWORD *)(v23 + 24);
  v24 = *(_BYTE *)v23;
  if ( *(_BYTE *)v23 )
  {
    if ( v24 == 22 )
    {
      v23 = *(_QWORD *)(v23 + 24);
    }
    else if ( v24 <= 0x1Cu )
    {
      v23 = 0;
    }
    else
    {
      v23 = sub_B43CB0(v23);
    }
  }
  v56 = 0;
  v55 = v23 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v25 = sub_26A73D0(a2, v23 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 0, 1);
  if ( !v25 || !*(_BYTE *)(v25 + 401) )
  {
LABEL_30:
    if ( *(_BYTE *)(a1 + 112) )
    {
LABEL_31:
      *(_QWORD *)(a1 + 104) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 104) = 0;
      *(_BYTE *)(a1 + 112) = 1;
    }
    goto LABEL_32;
  }
  if ( *(_BYTE *)(v25 + 337) )
  {
    v27 = *(unsigned int *)(v25 + 384);
    v26 = 1;
    if ( !(_DWORD)v27 )
      goto LABEL_34;
    v49 = v20;
    v50 = 0;
    v28 = *(__int64 **)(v25 + 376);
    v29 = &v28[v27];
    do
    {
      v30 = *v28;
      v56 = 0;
      v55 = v30 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
      v31 = sub_26A73D0(a2, v55, 0, a1, 0, 1);
      if ( !v31 || !*(_BYTE *)(v31 + 241) )
        goto LABEL_49;
      if ( *(_BYTE *)(v31 + 240) )
        LODWORD(v50) = v50 + 1;
      else
        ++HIDWORD(v50);
      ++v28;
    }
    while ( v29 != v28 );
    v38 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v38 = *(_QWORD *)(v38 + 24);
    v39 = (_QWORD *)sub_BD5C60(v38);
    if ( v50 )
    {
      v40 = sub_BCB2B0(v39);
      v41 = sub_ACD640(v40, 1, 0);
      v32 = *(_BYTE *)(a1 + 112) == 0;
      *(_QWORD *)(a1 + 104) = v41;
      if ( !v32 )
      {
LABEL_80:
        if ( !v49 )
          goto LABEL_33;
        v26 = *(_QWORD *)(a1 + 104) == v53;
        goto LABEL_34;
      }
    }
    else
    {
      v44 = sub_BCB2B0(v39);
      v45 = sub_ACD640(v44, 0, 0);
      v32 = *(_BYTE *)(a1 + 112) == 0;
      *(_QWORD *)(a1 + 104) = v45;
      if ( !v32 )
        goto LABEL_80;
    }
    *(_BYTE *)(a1 + 112) = 1;
    goto LABEL_80;
  }
LABEL_49:
  v32 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = 0;
  if ( v32 )
    *(_BYTE *)(a1 + 112) = 1;
LABEL_32:
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
LABEL_33:
  v26 = 0;
LABEL_34:
  sub_250C0C0((int *)&v54, v26);
  return v54;
}
