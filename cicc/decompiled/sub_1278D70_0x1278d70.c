// Function: sub_1278D70
// Address: 0x1278d70
//
unsigned __int64 __fastcall sub_1278D70(_QWORD *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  char v8; // cl
  unsigned __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 v13; // r10
  __int64 v14; // r8
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 j; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  char v22; // dl
  unsigned __int64 v23; // r12
  __int64 v25; // rax
  __int64 v26; // rdi
  _BOOL4 v27; // eax
  bool v28; // zf
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rcx
  size_t v35; // rdx
  size_t v36; // rax
  int v37; // esi
  int v38; // esi
  __int64 v39; // r8
  unsigned int v40; // edx
  int v41; // ecx
  __int64 v42; // rdi
  int v43; // r11d
  __int64 *v44; // r9
  __int64 v45; // rax
  _BYTE *v46; // rsi
  __int64 k; // rax
  __int64 v48; // r15
  unsigned __int64 v49; // rax
  __int64 *v50; // r11
  int v51; // ecx
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  int v55; // r11d
  unsigned int v56; // edx
  __int64 v57; // rdi
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  unsigned int v60; // [rsp+0h] [rbp-60h]
  int v61; // [rsp+8h] [rbp-58h]
  int v62; // [rsp+8h] [rbp-58h]
  unsigned int v63; // [rsp+8h] [rbp-58h]
  char i; // [rsp+10h] [rbp-50h]
  unsigned __int64 v65; // [rsp+10h] [rbp-50h]
  _QWORD v67[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a2;
  for ( i = sub_127B460(a2); *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  sub_1643330(*(_QWORD *)*a1);
  v7 = *(_QWORD *)(v6 + 160);
  if ( !v7 )
  {
    v23 = *(_QWORD *)(v6 + 128);
    v9 = v23;
    if ( !v23 )
      return v9;
    goto LABEL_63;
  }
  v8 = 0;
  v9 = 0;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v7 + 146) & 8) != 0 )
      goto LABEL_22;
    if ( (*(_BYTE *)(v7 + 144) & 4) != 0 )
      break;
    if ( *(_QWORD *)(v7 + 128) != v9 )
    {
      if ( *(_QWORD *)(v7 + 128) < v9 )
        sub_127B550("internal error during structure layout!");
      if ( v8 || i || (unsigned __int8)sub_12789C0((__int64)a1, *(_QWORD *)(v7 + 120)) )
      {
        v10 = *(_QWORD *)(v7 + 128);
      }
      else
      {
        v31 = sub_127A050(a1, *(_QWORD *)(v7 + 120));
        v32 = sub_15A9FE0(a1[2], v31);
        v10 = *(_QWORD *)(v7 + 128);
        if ( v10 == v32 * ((v32 + v9 - 1) / v32) )
        {
          v9 = *(_QWORD *)(v7 + 128);
          goto LABEL_13;
        }
      }
      v61 = v10 - v9;
      sub_12781D0(v10 - v9, a3, *(_QWORD *)*a1);
      *a4 += v61;
      v9 = *(_QWORD *)(v7 + 128);
    }
LABEL_13:
    if ( *(_QWORD *)(v6 + 8)
      && (v11 = *(_QWORD *)(v7 + 120), (*(_BYTE *)(v11 + 140) & 0xFB) == 8)
      && (sub_8D4C10(v11, dword_4F077C4 != 2) & 4) != 0 )
    {
      v33 = *(_QWORD *)(v6 + 8);
      v34 = *a4;
      v35 = 0;
      if ( v33 )
      {
        v60 = *a4;
        v36 = strlen(*(const char **)(v6 + 8));
        v34 = v60;
        v35 = v36;
      }
      sub_1CCC710(*a1, v33, v35, v34);
      v12 = *((_DWORD *)a1 + 32);
      v13 = (__int64)(a1 + 13);
      if ( !v12 )
      {
LABEL_45:
        ++a1[13];
        goto LABEL_46;
      }
    }
    else
    {
      v12 = *((_DWORD *)a1 + 32);
      v13 = (__int64)(a1 + 13);
      if ( !v12 )
        goto LABEL_45;
    }
    v14 = a1[14];
    v15 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( *v16 != v7 )
    {
      v62 = 1;
      v50 = 0;
      while ( v17 != -8 )
      {
        if ( !v50 && v17 == -16 )
          v50 = v16;
        v15 = (v12 - 1) & (v62 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == v7 )
          goto LABEL_17;
        ++v62;
      }
      v51 = *((_DWORD *)a1 + 30);
      if ( v50 )
        v16 = v50;
      ++a1[13];
      v41 = v51 + 1;
      if ( 4 * v41 >= 3 * v12 )
      {
LABEL_46:
        sub_1278480(v13, 2 * v12);
        v37 = *((_DWORD *)a1 + 32);
        if ( !v37 )
          goto LABEL_110;
        v38 = v37 - 1;
        v39 = a1[14];
        v40 = v38 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v41 = *((_DWORD *)a1 + 30) + 1;
        v16 = (__int64 *)(v39 + 16LL * v40);
        v42 = *v16;
        if ( *v16 != v7 )
        {
          v43 = 1;
          v44 = 0;
          while ( v42 != -8 )
          {
            if ( v42 == -16 && !v44 )
              v44 = v16;
            v40 = v38 & (v43 + v40);
            v16 = (__int64 *)(v39 + 16LL * v40);
            v42 = *v16;
            if ( *v16 == v7 )
              goto LABEL_85;
            ++v43;
          }
          goto LABEL_92;
        }
      }
      else if ( v12 - *((_DWORD *)a1 + 31) - v41 <= v12 >> 3 )
      {
        v63 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
        sub_1278480(v13, v12);
        v52 = *((_DWORD *)a1 + 32);
        if ( !v52 )
        {
LABEL_110:
          ++*((_DWORD *)a1 + 30);
          BUG();
        }
        v53 = v52 - 1;
        v54 = a1[14];
        v55 = 1;
        v44 = 0;
        v56 = v53 & v63;
        v41 = *((_DWORD *)a1 + 30) + 1;
        v16 = (__int64 *)(v54 + 16LL * (v53 & v63));
        v57 = *v16;
        if ( *v16 != v7 )
        {
          while ( v57 != -8 )
          {
            if ( !v44 && v57 == -16 )
              v44 = v16;
            v56 = v53 & (v55 + v56);
            v16 = (__int64 *)(v54 + 16LL * v56);
            v57 = *v16;
            if ( *v16 == v7 )
              goto LABEL_85;
            ++v55;
          }
LABEL_92:
          if ( v44 )
            v16 = v44;
        }
      }
LABEL_85:
      *((_DWORD *)a1 + 30) = v41;
      if ( *v16 != -8 )
        --*((_DWORD *)a1 + 31);
      *v16 = v7;
      *((_DWORD *)v16 + 2) = 0;
    }
LABEL_17:
    *((_DWORD *)v16 + 2) = *a4;
    if ( (*(_BYTE *)(v7 + 145) & 0x10) != 0 )
    {
      LODWORD(v67[0]) = 0;
      for ( j = *(_QWORD *)(v7 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( (*(_BYTE *)(v7 + 146) & 4) != 0 )
      {
        v25 = *(_QWORD *)(j + 168);
        if ( (*(_BYTE *)(v25 + 109) & 0x10) != 0 )
          j = *(_QWORD *)(v25 + 208);
      }
      v19 = sub_1278D70(a1, j, a3, v67);
      v8 = 0;
      v9 += v19;
      *a4 += LODWORD(v67[0]);
    }
    else
    {
      v45 = sub_127A050(a1, *(_QWORD *)(v7 + 120));
      v67[0] = v45;
      v46 = *(_BYTE **)(a3 + 8);
      if ( v46 == *(_BYTE **)(a3 + 16) )
      {
        sub_1277EB0(a3, v46, v67);
      }
      else
      {
        if ( v46 )
        {
          *(_QWORD *)v46 = v45;
          v46 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v46 + 8;
      }
      ++*a4;
      for ( k = *(_QWORD *)(v7 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v9 += *(_QWORD *)(k + 128);
      v8 = 0;
    }
LABEL_22:
    v20 = *(_QWORD *)(v7 + 112);
    if ( !v20 )
      goto LABEL_25;
LABEL_23:
    v7 = v20;
  }
  v20 = *(_QWORD *)(v7 + 112);
  v8 = 1;
  if ( v20 )
    goto LABEL_23;
LABEL_25:
  v65 = v9;
  v21 = v9;
  v22 = *(_BYTE *)(v7 + 144) & 4;
  if ( v22 )
  {
    v9 = *(_QWORD *)(v6 + 128);
    v23 = v9 - v21;
    goto LABEL_27;
  }
  v26 = *(_QWORD *)(v7 + 120);
  if ( *(_BYTE *)(v6 + 140) == 11 )
  {
    v29 = *(_BYTE *)(v26 + 140);
    if ( v29 == 12 )
      goto LABEL_66;
    goto LABEL_68;
  }
  v27 = sub_8D3410(v26);
  v26 = *(_QWORD *)(v7 + 120);
  v28 = !v27;
  v29 = *(_BYTE *)(v26 + 140);
  if ( v28 )
  {
    v22 = *(_BYTE *)(v7 + 144) & 4;
    if ( v29 == 12 )
      goto LABEL_66;
    goto LABEL_67;
  }
  if ( v29 != 12 )
  {
    if ( *(_QWORD *)(v26 + 128) )
    {
      v22 = *(_BYTE *)(v7 + 144) & 4;
      goto LABEL_67;
    }
    return v9;
  }
  v30 = *(_QWORD *)(v7 + 120);
  do
    v30 = *(_QWORD *)(v30 + 160);
  while ( *(_BYTE *)(v30 + 140) == 12 );
  if ( !*(_QWORD *)(v30 + 128) )
    return v9;
  v22 = *(_BYTE *)(v7 + 144) & 4;
  do
  {
LABEL_66:
    v26 = *(_QWORD *)(v26 + 160);
    v29 = *(_BYTE *)(v26 + 140);
  }
  while ( v29 == 12 );
LABEL_67:
  if ( v22 )
  {
    v49 = *(_QWORD *)(v7 + 128)
        + (*(unsigned __int8 *)(v7 + 136) + *(_QWORD *)(v7 + 176) + (unsigned __int64)(dword_4F06BA0 - 1))
        / dword_4F06BA0;
  }
  else
  {
LABEL_68:
    v48 = *(_QWORD *)(v7 + 128);
    if ( (*(_BYTE *)(v7 + 146) & 4) != 0 && (unsigned __int8)(v29 - 9) <= 1u )
    {
      v58 = sub_730E80(v26);
      v59 = *(_QWORD *)(v26 + 168);
      if ( *(_QWORD *)(v59 + 32) >= v58 )
        v58 = *(_QWORD *)(v59 + 32);
      v49 = v48 + v58;
    }
    else
    {
      v49 = v48 + *(_QWORD *)(v26 + 128);
    }
  }
  v23 = *(_QWORD *)(v6 + 128);
  if ( *(_BYTE *)(v6 + 140) == 11 )
  {
    v9 = v65;
    if ( v49 < v23 )
      goto LABEL_73;
    return v9;
  }
  v9 = v65;
  v23 -= v49;
LABEL_73:
  v9 += v23;
LABEL_27:
  if ( v23 )
  {
LABEL_63:
    sub_12781D0(v23, a3, *(_QWORD *)*a1);
    *a4 += v23;
  }
  return v9;
}
