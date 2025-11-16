// Function: sub_1A122A0
// Address: 0x1a122a0
//
char __fastcall sub_1A122A0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 *v10; // rax
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r15
  int v20; // r9d
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // r8
  unsigned int v26; // r9d
  unsigned __int64 v27; // r15
  unsigned int v28; // r14d
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // r14
  int v35; // r11d
  __int64 *v36; // r10
  int v37; // eax
  int v38; // ecx
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  int v42; // esi
  __int64 v43; // rdi
  unsigned int v44; // eax
  __int64 v45; // r8
  int v46; // r10d
  __int64 *v47; // r9
  int v48; // eax
  int v49; // eax
  __int64 v50; // rdi
  __int64 *v51; // r8
  unsigned int v52; // r15d
  int v53; // r9d
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v57; // [rsp+8h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-38h]
  __int64 v59; // [rsp+18h] [rbp-38h]
  __int64 v60; // [rsp+18h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *sub_1A10F60(a1, *v7);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v10 = sub_1A10F60(a1, *(_QWORD *)(v9 + 24));
  v11 = *(_DWORD *)(a1 + 144);
  v12 = *v10;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_70;
  }
  v13 = *(_QWORD *)(a1 + 128);
  v14 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( a2 != *v15 )
  {
    v35 = 1;
    v36 = 0;
    while ( v16 != -8 )
    {
      if ( !v36 && v16 == -16 )
        v36 = v15;
      v14 = (v11 - 1) & (v35 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_7;
      ++v35;
    }
    v37 = *(_DWORD *)(a1 + 136);
    if ( v36 )
      v15 = v36;
    ++*(_QWORD *)(a1 + 120);
    v38 = v37 + 1;
    if ( 4 * (v37 + 1) < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a1 + 140) - v38 > v11 >> 3 )
      {
LABEL_57:
        *(_DWORD *)(a1 + 136) = v38;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 140);
        *v15 = a2;
        v15[1] = 0;
        goto LABEL_8;
      }
      v60 = v12;
      sub_1A0FE70(a1 + 120, v11);
      v48 = *(_DWORD *)(a1 + 144);
      if ( v48 )
      {
        v49 = v48 - 1;
        v50 = *(_QWORD *)(a1 + 128);
        v51 = 0;
        v52 = v49 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v53 = 1;
        v38 = *(_DWORD *)(a1 + 136) + 1;
        v12 = v60;
        v15 = (__int64 *)(v50 + 16LL * v52);
        v54 = *v15;
        if ( a2 != *v15 )
        {
          while ( v54 != -8 )
          {
            if ( v54 == -16 && !v51 )
              v51 = v15;
            v52 = v49 & (v53 + v52);
            v15 = (__int64 *)(v50 + 16LL * v52);
            v54 = *v15;
            if ( a2 == *v15 )
              goto LABEL_57;
            ++v53;
          }
          if ( v51 )
            v15 = v51;
        }
        goto LABEL_57;
      }
LABEL_110:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_70:
    v59 = v12;
    sub_1A0FE70(a1 + 120, 2 * v11);
    v41 = *(_DWORD *)(a1 + 144);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(a1 + 128);
      v44 = (v41 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v38 = *(_DWORD *)(a1 + 136) + 1;
      v12 = v59;
      v15 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v15;
      if ( a2 != *v15 )
      {
        v46 = 1;
        v47 = 0;
        while ( v45 != -8 )
        {
          if ( !v47 && v45 == -16 )
            v47 = v15;
          v44 = v42 & (v46 + v44);
          v15 = (__int64 *)(v43 + 16LL * v44);
          v45 = *v15;
          if ( a2 == *v15 )
            goto LABEL_57;
          ++v46;
        }
        if ( v47 )
          v15 = v47;
      }
      goto LABEL_57;
    }
    goto LABEL_110;
  }
LABEL_7:
  v17 = v15[1] ^ 6;
  if ( (v17 & 6) == 0 )
    return v17;
LABEL_8:
  v18 = v8;
  v19 = (v8 >> 1) & 3;
  v20 = v19;
  LOBYTE(v17) = (_DWORD)v19 == 2 || v19 == 1;
  if ( (_BYTE)v17 && ((v21 = (v12 >> 1) & 3, v21 == 2) || v21 == 1) )
  {
    v17 = sub_15A2A30(
            (__int64 *)((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24),
            (__int64 *)(v8 & 0xFFFFFFFFFFFFFFF8LL),
            v12 & 0xFFFFFFFFFFFFFFF8LL,
            0,
            0,
            a3,
            a4,
            a5);
    if ( *(_BYTE *)(v17 + 16) == 9 )
      return v17;
    v31 = v15[1];
    v32 = (v31 >> 1) & 3;
    if ( v32 == 1 || v32 == 3 )
      return v17;
    if ( (_DWORD)v32 )
    {
      if ( v17 == (v31 & 0xFFFFFFFFFFFFFFF8LL) )
        return v17;
      LOBYTE(v33) = v31 | 6;
      v15[1] = v31 | 6;
    }
    else
    {
      v33 = v31 & 1 | v17 | 2;
      v15[1] = v33;
    }
  }
  else
  {
    if ( v19 != 3 && (((unsigned __int8)v12 ^ 6) & 6) != 0 )
      return v17;
    v22 = *(unsigned __int8 *)(a2 + 16);
    v23 = (unsigned int)(v22 - 41);
    if ( (unsigned __int8)(v22 - 41) > 1u )
      goto LABEL_17;
    v57 = v12;
    v58 = v8;
    if ( !(_BYTE)v17 )
      goto LABEL_46;
    v24 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !sub_1593BB0(v24, v23, v12, v22) )
    {
      v22 = *(unsigned __int8 *)(a2 + 16);
      v18 = v58;
      v20 = v19;
      v12 = v57;
LABEL_17:
      LOBYTE(v17) = (_BYTE)v22 == 50 || (_BYTE)v22 == 39;
      if ( (_BYTE)v22 != 51 && !(_BYTE)v17 )
        goto LABEL_46;
      if ( v19 == 3 )
      {
        v18 = v12;
        v19 = (v12 >> 1) & 3;
        if ( v19 == 3 )
          goto LABEL_46;
        v20 = (v12 >> 1) & 3;
      }
      if ( !v19 )
        return v17;
      if ( (_BYTE)v17 )
      {
        v34 = v18 & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v23) = (_BYTE)v22 == 50;
        if ( sub_1593BB0(v18 & 0xFFFFFFFFFFFFFFF8LL, v23, v12, v22) )
        {
          v17 = v15[1];
          v39 = (v17 >> 1) & 3;
          if ( v39 == 3 || v39 == 1 )
            return v17;
          if ( (_DWORD)v39 )
          {
            if ( v34 == (v17 & 0xFFFFFFFFFFFFFFF8LL) )
              return v17;
            v40 = v17 | 6;
            v15[1] = v40;
          }
          else
          {
            v40 = v34 | v17 & 1 | 2;
            v15[1] = v40;
          }
          if ( (((unsigned __int8)v40 ^ 6) & 6) != 0 )
            goto LABEL_31;
LABEL_66:
          if ( *(_DWORD *)(a1 + 824) >= *(_DWORD *)(a1 + 828) )
            sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, v25, v26);
          v17 = *(_QWORD *)(a1 + 816);
          *(_QWORD *)(v17 + 8LL * (unsigned int)(*(_DWORD *)(a1 + 824))++) = a2;
          return v17;
        }
      }
      else
      {
        v26 = v20 - 1;
        if ( v26 <= 1 )
        {
          v25 = v18 & 0xFFFFFFFFFFFFFFF8LL;
          v27 = v25;
          if ( *(_BYTE *)(v25 + 16) == 13 )
          {
            v28 = *(_DWORD *)(v25 + 32);
            if ( v28 <= 0x40 )
            {
              if ( *(_QWORD *)(v25 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v28) )
                goto LABEL_26;
            }
            else if ( v28 == (unsigned int)sub_16A58F0(v25 + 24) )
            {
LABEL_26:
              v29 = v15[1];
              v17 = (v29 >> 1) & 3;
              if ( v17 == 1 || v17 == 3 )
                return v17;
              if ( (_DWORD)v17 )
              {
                LOBYTE(v17) = v29 & 0xF8;
                if ( v27 == (v29 & 0xFFFFFFFFFFFFFFF8LL) )
                  return v17;
                v30 = v29 | 6;
                v15[1] = v30;
              }
              else
              {
                v30 = v27 | v29 & 1 | 2;
                v15[1] = v30;
              }
              if ( (((unsigned __int8)v30 ^ 6) & 6) != 0 )
                goto LABEL_31;
              goto LABEL_66;
            }
          }
        }
      }
LABEL_46:
      LOBYTE(v17) = sub_1A11830(a1, a2);
      return v17;
    }
    v17 = v15[1];
    v55 = (v17 >> 1) & 3;
    if ( v55 == 1 || v55 == 3 )
      return v17;
    if ( (_DWORD)v55 )
    {
      if ( v24 == (v17 & 0xFFFFFFFFFFFFFFF8LL) )
        return v17;
      v33 = v17 | 6;
      v15[1] = v33;
    }
    else
    {
      v33 = v24 | v17 & 1 | 2;
      v15[1] = v33;
    }
  }
  if ( (((unsigned __int8)v33 ^ 6) & 6) != 0 )
  {
LABEL_31:
    v17 = *(unsigned int *)(a1 + 1352);
    if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 1356) )
    {
      sub_16CD150(a1 + 1344, (const void *)(a1 + 1360), 0, 8, v25, v26);
      v17 = *(unsigned int *)(a1 + 1352);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * v17) = a2;
    ++*(_DWORD *)(a1 + 1352);
    return v17;
  }
  v17 = *(unsigned int *)(a1 + 824);
  if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 828) )
  {
    sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, v25, v26);
    v17 = *(unsigned int *)(a1 + 824);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * v17) = a2;
  ++*(_DWORD *)(a1 + 824);
  return v17;
}
