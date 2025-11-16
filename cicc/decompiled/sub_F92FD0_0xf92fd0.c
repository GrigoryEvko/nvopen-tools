// Function: sub_F92FD0
// Address: 0xf92fd0
//
bool __fastcall sub_F92FD0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // r14
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // rbx
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rsi
  __int64 *v22; // r9
  _QWORD *v23; // rcx
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // rsi
  char v28; // al
  unsigned __int8 **v29; // rax
  __int64 v30; // rdx
  unsigned __int8 **v31; // rdi
  __int64 v32; // rcx
  unsigned __int8 **v33; // rsi
  unsigned __int8 *v34; // rcx
  unsigned __int8 **v35; // r8
  unsigned __int8 v36; // dl
  unsigned __int8 *v37; // rcx
  unsigned __int8 v38; // dl
  unsigned __int8 *v39; // rcx
  unsigned __int8 v40; // dl
  unsigned __int8 v41; // dl
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rax
  int v45; // edi
  __int64 v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  unsigned __int8 v49; // dl
  unsigned __int8 v50; // dl
  unsigned __int8 v51; // dl
  unsigned int v52; // [rsp+Ch] [rbp-54h]
  __int64 *v53; // [rsp+10h] [rbp-50h]
  __int64 v54; // [rsp+18h] [rbp-48h]

  v2 = *(__int64 **)(a2 + 16);
  v3 = *v2;
  v4 = 32LL * (*(_DWORD *)(*v2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
  {
    v5 = *(_QWORD *)(v3 - 8);
    v54 = v5 + v4;
  }
  else
  {
    v54 = v3;
    v5 = v3 - v4;
  }
  if ( v5 == v54 )
    return 1;
  v52 = 0;
  while ( 1 )
  {
    v6 = *(_QWORD *)(*a1 + 8LL);
    v7 = *(unsigned int *)(*a1 + 24LL);
    if ( !(_DWORD)v7 )
      goto LABEL_29;
    v8 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v9 = v6 + 56LL * v8;
    v10 = *(_QWORD *)v9;
    if ( *(_QWORD *)v9 != v5 )
    {
      v45 = 1;
      while ( v10 != -4096 )
      {
        v8 = (v7 - 1) & (v45 + v8);
        v9 = v6 + 56LL * v8;
        v10 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 == v5 )
          goto LABEL_7;
        ++v45;
      }
      goto LABEL_29;
    }
LABEL_7:
    v11 = 7 * v7;
    if ( v9 == v6 + 56 * v7 )
      goto LABEL_29;
    v12 = *(__int64 **)(v9 + 8);
    v13 = 8LL * *(unsigned int *)(v9 + 16);
    v14 = a1[1];
    v53 = &v12[(unsigned __int64)v13 / 8];
    v15 = v13 >> 3;
    v16 = v13 >> 5;
    if ( v16 )
    {
      v17 = &v12[4 * v16];
      while ( 1 )
      {
        v18 = *v12;
        if ( *(_BYTE *)(v14 + 28) )
          break;
        if ( !sub_C8CA60(v14, v18) )
          goto LABEL_32;
        v21 = v12[1];
        v22 = v12 + 1;
        if ( *(_BYTE *)(v14 + 28) )
        {
          v19 = *(_QWORD **)(v14 + 8);
          v11 = (__int64)&v19[*(unsigned int *)(v14 + 20)];
          if ( v19 != (_QWORD *)v11 )
          {
            v23 = *(_QWORD **)(v14 + 8);
LABEL_17:
            while ( *v19 != v21 )
            {
              if ( (_QWORD *)v11 == ++v19 )
                goto LABEL_31;
            }
            v24 = v12[2];
            v22 = v12 + 2;
            v25 = v23;
            do
            {
LABEL_20:
              if ( v24 == *v23 )
              {
                v26 = v12[3];
                v10 = (__int64)(v12 + 3);
                goto LABEL_23;
              }
              ++v23;
            }
            while ( (_QWORD *)v11 != v23 );
          }
          goto LABEL_31;
        }
        v42 = sub_C8CA60(v14, v21);
        v22 = v12 + 1;
        if ( !v42 )
          goto LABEL_31;
        v24 = v12[2];
        v22 = v12 + 2;
        if ( *(_BYTE *)(v14 + 28) )
        {
          v23 = *(_QWORD **)(v14 + 8);
          v11 = (__int64)&v23[*(unsigned int *)(v14 + 20)];
          if ( v23 != (_QWORD *)v11 )
          {
            v25 = *(_QWORD **)(v14 + 8);
            goto LABEL_20;
          }
LABEL_31:
          v12 = v22;
          goto LABEL_32;
        }
        v43 = sub_C8CA60(v14, v24);
        v22 = v12 + 2;
        if ( !v43 )
          goto LABEL_31;
        v26 = v12[3];
        v10 = (__int64)(v12 + 3);
        if ( *(_BYTE *)(v14 + 28) )
        {
          v25 = *(_QWORD **)(v14 + 8);
          v11 = (__int64)&v25[*(unsigned int *)(v14 + 20)];
          if ( v25 == (_QWORD *)v11 )
            goto LABEL_62;
LABEL_23:
          while ( *v25 != v26 )
          {
            if ( ++v25 == (_QWORD *)v11 )
              goto LABEL_62;
          }
          v12 += 4;
          if ( v17 == v12 )
          {
LABEL_25:
            v15 = v53 - v12;
            goto LABEL_26;
          }
        }
        else
        {
          v44 = sub_C8CA60(v14, v26);
          v10 = (__int64)(v12 + 3);
          if ( !v44 )
          {
LABEL_62:
            v12 = (__int64 *)v10;
            goto LABEL_32;
          }
          v12 += 4;
          if ( v17 == v12 )
            goto LABEL_25;
        }
      }
      v19 = *(_QWORD **)(v14 + 8);
      v11 = (__int64)&v19[*(unsigned int *)(v14 + 20)];
      if ( v19 != (_QWORD *)v11 )
      {
        v20 = *(_QWORD **)(v14 + 8);
        do
        {
          if ( v18 == *v20 )
          {
            v21 = v12[1];
            v22 = v12 + 1;
            v23 = *(_QWORD **)(v14 + 8);
            goto LABEL_17;
          }
          ++v20;
        }
        while ( (_QWORD *)v11 != v20 );
      }
      goto LABEL_32;
    }
LABEL_26:
    if ( v15 == 2 )
      goto LABEL_72;
    if ( v15 == 3 )
    {
      if ( !(unsigned __int8)sub_B19060(v14, *v12, v11, v10) )
        goto LABEL_32;
      ++v12;
LABEL_72:
      if ( !(unsigned __int8)sub_B19060(v14, *v12, v11, v10) )
        goto LABEL_32;
      ++v12;
      goto LABEL_74;
    }
    if ( v15 != 1 )
      goto LABEL_29;
LABEL_74:
    v46 = *v12;
    if ( *(_BYTE *)(v14 + 28) )
    {
      v47 = *(_QWORD **)(v14 + 8);
      v48 = &v47[*(unsigned int *)(v14 + 20)];
      if ( v47 != v48 )
      {
        while ( v46 != *v47 )
        {
          if ( v48 == ++v47 )
            goto LABEL_32;
        }
        goto LABEL_29;
      }
    }
    else if ( sub_C8CA60(v14, v46) )
    {
      goto LABEL_29;
    }
LABEL_32:
    if ( v53 == v12 )
      goto LABEL_29;
    ++v52;
    v28 = **(_BYTE **)(v5 + 24);
    if ( v28 == 61 )
    {
      if ( (unsigned int)sub_BD2910(v5) )
        goto LABEL_29;
    }
    else if ( v28 != 62 || (unsigned int)sub_BD2910(v5) != 1 )
    {
      goto LABEL_29;
    }
    v29 = *(unsigned __int8 ***)(v9 + 8);
    v30 = 8LL * *(unsigned int *)(v9 + 16);
    v31 = &v29[(unsigned __int64)v30 / 8];
    v32 = v30 >> 3;
    if ( v30 >> 5 )
      break;
LABEL_93:
    if ( v32 != 2 )
    {
      if ( v32 != 3 )
      {
        if ( v32 != 1 )
          goto LABEL_29;
        goto LABEL_96;
      }
      v50 = **v29;
      if ( v50 > 0x1Cu )
      {
        if ( v50 == 63 )
          goto LABEL_49;
      }
      else if ( v50 == 5 && *((_WORD *)*v29 + 1) == 34 )
      {
LABEL_49:
        if ( v31 != v29 )
          return 0;
        goto LABEL_29;
      }
      ++v29;
    }
    v51 = **v29;
    if ( v51 <= 0x1Cu )
    {
      if ( v51 == 5 && *((_WORD *)*v29 + 1) == 34 )
        goto LABEL_49;
    }
    else if ( v51 == 63 )
    {
      goto LABEL_49;
    }
    ++v29;
LABEL_96:
    v49 = **v29;
    if ( v49 <= 0x1Cu )
    {
      if ( v49 == 5 && *((_WORD *)*v29 + 1) == 34 )
        goto LABEL_49;
    }
    else if ( v49 == 63 )
    {
      goto LABEL_49;
    }
LABEL_29:
    v5 += 32;
    if ( v54 == v5 )
      return v52 <= 1;
  }
  v33 = &v29[4 * (v30 >> 5)];
  while ( 1 )
  {
    v41 = **v29;
    if ( v41 > 0x1Cu )
    {
      if ( v41 == 63 )
        goto LABEL_49;
    }
    else if ( v41 == 5 && *((_WORD *)*v29 + 1) == 34 )
    {
      goto LABEL_49;
    }
    v34 = v29[1];
    v35 = v29 + 1;
    v36 = *v34;
    if ( *v34 <= 0x1Cu )
      break;
    if ( v36 == 63 )
      goto LABEL_84;
LABEL_41:
    v37 = v29[2];
    v35 = v29 + 2;
    v38 = *v37;
    if ( *v37 > 0x1Cu )
    {
      if ( v38 == 63 )
        goto LABEL_84;
LABEL_43:
      v39 = v29[3];
      v35 = v29 + 3;
      v40 = *v39;
      if ( *v39 > 0x1Cu )
        goto LABEL_44;
      goto LABEL_89;
    }
    if ( v38 != 5 )
      goto LABEL_43;
    if ( *((_WORD *)v37 + 1) == 34 )
      goto LABEL_84;
    v39 = v29[3];
    v35 = v29 + 3;
    v40 = *v39;
    if ( *v39 > 0x1Cu )
    {
LABEL_44:
      if ( v40 == 63 )
        goto LABEL_84;
      goto LABEL_45;
    }
LABEL_89:
    if ( v40 == 5 )
    {
      if ( *((_WORD *)v39 + 1) == 34 )
        goto LABEL_84;
      v29 += 4;
      if ( v33 == v29 )
      {
LABEL_92:
        v32 = v31 - v29;
        goto LABEL_93;
      }
    }
    else
    {
LABEL_45:
      v29 += 4;
      if ( v33 == v29 )
        goto LABEL_92;
    }
  }
  if ( v36 != 5 || *((_WORD *)v34 + 1) != 34 )
    goto LABEL_41;
LABEL_84:
  if ( v31 == v35 )
    goto LABEL_29;
  return 0;
}
