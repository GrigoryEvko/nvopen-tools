// Function: sub_919130
// Address: 0x919130
//
unsigned __int64 __fastcall sub_919130(_QWORD *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  char v8; // r14
  unsigned __int64 v9; // rax
  int v10; // r14d
  __int64 v11; // rdi
  unsigned int v12; // esi
  unsigned int v13; // ecx
  __int64 v14; // r9
  unsigned int v15; // r8d
  _QWORD *v16; // rax
  __int64 v17; // rdi
  unsigned int *v18; // rax
  __int64 j; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  char v22; // dl
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r12
  __int64 v26; // rdi
  _BOOL4 v27; // eax
  bool v28; // zf
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // cl
  __int64 v33; // r14
  __int64 v34; // rcx
  size_t v35; // rdx
  size_t v36; // rax
  __int64 v37; // rax
  _BYTE *v38; // rsi
  __int64 k; // rax
  __int64 v40; // r14
  unsigned __int64 v41; // rax
  _QWORD *v42; // rdx
  int v43; // eax
  int v44; // eax
  int v45; // edi
  int v46; // edi
  __int64 v47; // r9
  unsigned int v48; // esi
  __int64 v49; // r8
  int v50; // r14d
  _QWORD *v51; // r11
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  _QWORD *v55; // r10
  unsigned int v56; // r14d
  int v57; // r11d
  __int64 v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  unsigned int v61; // [rsp+14h] [rbp-5Ch]
  int v62; // [rsp+14h] [rbp-5Ch]
  unsigned int v63; // [rsp+14h] [rbp-5Ch]
  unsigned int v64; // [rsp+14h] [rbp-5Ch]
  char i; // [rsp+18h] [rbp-58h]
  unsigned __int64 v67; // [rsp+28h] [rbp-48h]
  _QWORD v68[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a2;
  for ( i = sub_91B7B0(a2); *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  sub_BCB2B0(*(_QWORD *)*a1);
  v7 = *(_QWORD *)(v6 + 160);
  if ( !v7 )
  {
    v24 = *(_QWORD *)(v6 + 128);
    v23 = v24;
    if ( !v24 )
      return v23;
    goto LABEL_55;
  }
  v67 = 0;
  v8 = 0;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v7 + 146) & 8) != 0 )
      goto LABEL_26;
    if ( (*(_BYTE *)(v7 + 144) & 4) != 0 )
      break;
    if ( *(_QWORD *)(v7 + 128) != v67 )
    {
      if ( *(_QWORD *)(v7 + 128) < v67 )
        sub_91B8A0("internal error during structure layout!");
      if ( i || v8 || (unsigned __int8)sub_918D80((__int64)a1, *(_QWORD *)(v7 + 120)) )
      {
        v9 = *(_QWORD *)(v7 + 128);
      }
      else
      {
        v31 = sub_91A3B0(a1, *(_QWORD *)(v7 + 120));
        v32 = sub_AE5020(a1[2], v31);
        v9 = *(_QWORD *)(v7 + 128);
        if ( v9 == ((v67 + (1LL << v32) - 1) & -(1LL << v32)) )
          goto LABEL_13;
      }
      v10 = v9 - v67;
      sub_918530(v9 - v67, a3, *(_QWORD *)*a1);
      *a4 += v10;
      v9 = *(_QWORD *)(v7 + 128);
LABEL_13:
      v67 = v9;
    }
    if ( *(_QWORD *)(v6 + 8) )
    {
      v11 = *(_QWORD *)(v7 + 120);
      if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 && (sub_8D4C10(v11, dword_4F077C4 != 2) & 4) != 0 )
      {
        v33 = *(_QWORD *)(v6 + 8);
        v34 = *a4;
        v35 = 0;
        if ( v33 )
        {
          v61 = *a4;
          v36 = strlen(*(const char **)(v6 + 8));
          v34 = v61;
          v35 = v36;
        }
        sub_CF0C10(*a1, v33, v35, v34);
      }
    }
    v12 = *((_DWORD *)a1 + 32);
    v13 = *a4;
    if ( !v12 )
    {
      ++a1[13];
      goto LABEL_82;
    }
    v14 = a1[14];
    v15 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v16 = (_QWORD *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v7 != *v16 )
    {
      v62 = 1;
      v42 = 0;
      while ( v17 != -4096 )
      {
        if ( v17 == -8192 && !v42 )
          v42 = v16;
        v15 = (v12 - 1) & (v62 + v15);
        v16 = (_QWORD *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v7 == *v16 )
          goto LABEL_18;
        ++v62;
      }
      if ( !v42 )
        v42 = v16;
      v43 = *((_DWORD *)a1 + 30);
      ++a1[13];
      v44 = v43 + 1;
      if ( 4 * v44 >= 3 * v12 )
      {
LABEL_82:
        v63 = v13;
        sub_9187E0((__int64)(a1 + 13), 2 * v12);
        v45 = *((_DWORD *)a1 + 32);
        if ( !v45 )
          goto LABEL_115;
        v46 = v45 - 1;
        v47 = a1[14];
        v13 = v63;
        v48 = v46 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v44 = *((_DWORD *)a1 + 30) + 1;
        v42 = (_QWORD *)(v47 + 16LL * v48);
        v49 = *v42;
        if ( v7 != *v42 )
        {
          v50 = 1;
          v51 = 0;
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v51 )
              v51 = v42;
            v48 = v46 & (v50 + v48);
            v42 = (_QWORD *)(v47 + 16LL * v48);
            v49 = *v42;
            if ( v7 == *v42 )
              goto LABEL_77;
            ++v50;
          }
          if ( v51 )
            v42 = v51;
        }
      }
      else if ( v12 - *((_DWORD *)a1 + 31) - v44 <= v12 >> 3 )
      {
        v64 = v13;
        sub_9187E0((__int64)(a1 + 13), v12);
        v52 = *((_DWORD *)a1 + 32);
        if ( !v52 )
        {
LABEL_115:
          ++*((_DWORD *)a1 + 30);
          BUG();
        }
        v53 = v52 - 1;
        v54 = a1[14];
        v55 = 0;
        v56 = v53 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v13 = v64;
        v57 = 1;
        v44 = *((_DWORD *)a1 + 30) + 1;
        v42 = (_QWORD *)(v54 + 16LL * v56);
        v58 = *v42;
        if ( v7 != *v42 )
        {
          while ( v58 != -4096 )
          {
            if ( !v55 && v58 == -8192 )
              v55 = v42;
            v56 = v53 & (v57 + v56);
            v42 = (_QWORD *)(v54 + 16LL * v56);
            v58 = *v42;
            if ( v7 == *v42 )
              goto LABEL_77;
            ++v57;
          }
          if ( v55 )
            v42 = v55;
        }
      }
LABEL_77:
      *((_DWORD *)a1 + 30) = v44;
      if ( *v42 != -4096 )
        --*((_DWORD *)a1 + 31);
      *v42 = v7;
      v18 = (unsigned int *)(v42 + 1);
      *((_DWORD *)v42 + 2) = 0;
      goto LABEL_19;
    }
LABEL_18:
    v18 = (unsigned int *)(v16 + 1);
LABEL_19:
    *v18 = v13;
    if ( (*(_BYTE *)(v7 + 145) & 0x10) != 0 )
    {
      LODWORD(v68[0]) = 0;
      for ( j = *(_QWORD *)(v7 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( (*(_BYTE *)(v7 + 146) & 4) != 0 )
      {
        v20 = *(_QWORD *)(j + 168);
        if ( (*(_BYTE *)(v20 + 109) & 0x10) != 0 )
          j = *(_QWORD *)(v20 + 208);
      }
      v8 = 0;
      v67 += sub_919130(a1, j, a3, v68);
      *a4 += LODWORD(v68[0]);
    }
    else
    {
      v37 = sub_91A3B0(a1, *(_QWORD *)(v7 + 120));
      v68[0] = v37;
      v38 = *(_BYTE **)(a3 + 8);
      if ( v38 == *(_BYTE **)(a3 + 16) )
      {
        sub_918210(a3, v38, v68);
      }
      else
      {
        if ( v38 )
        {
          *(_QWORD *)v38 = v37;
          v38 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v38 + 8;
      }
      ++*a4;
      for ( k = *(_QWORD *)(v7 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v8 = 0;
      v67 += *(_QWORD *)(k + 128);
    }
LABEL_26:
    v21 = *(_QWORD *)(v7 + 112);
    if ( !v21 )
      goto LABEL_29;
LABEL_27:
    v7 = v21;
  }
  v21 = *(_QWORD *)(v7 + 112);
  v8 = 1;
  if ( v21 )
    goto LABEL_27;
LABEL_29:
  v22 = *(_BYTE *)(v7 + 144) & 4;
  if ( v22 )
  {
    v23 = *(_QWORD *)(v6 + 128);
    v24 = v23 - v67;
    goto LABEL_31;
  }
  v26 = *(_QWORD *)(v7 + 120);
  if ( *(_BYTE *)(v6 + 140) == 11 )
  {
    v29 = *(_BYTE *)(v26 + 140);
    if ( v29 == 12 )
      goto LABEL_58;
    goto LABEL_60;
  }
  v27 = sub_8D3410(v26);
  v26 = *(_QWORD *)(v7 + 120);
  v28 = !v27;
  v29 = *(_BYTE *)(v26 + 140);
  if ( v28 )
  {
    v22 = *(_BYTE *)(v7 + 144) & 4;
    if ( v29 == 12 )
      goto LABEL_58;
    goto LABEL_59;
  }
  if ( v29 != 12 )
  {
    v23 = v67;
    if ( *(_QWORD *)(v26 + 128) )
    {
      v22 = *(_BYTE *)(v7 + 144) & 4;
      goto LABEL_59;
    }
    return v23;
  }
  v30 = *(_QWORD *)(v7 + 120);
  do
    v30 = *(_QWORD *)(v30 + 160);
  while ( *(_BYTE *)(v30 + 140) == 12 );
  v23 = v67;
  if ( !*(_QWORD *)(v30 + 128) )
    return v23;
  v22 = *(_BYTE *)(v7 + 144) & 4;
  do
  {
LABEL_58:
    v26 = *(_QWORD *)(v26 + 160);
    v29 = *(_BYTE *)(v26 + 140);
  }
  while ( v29 == 12 );
LABEL_59:
  if ( v22 )
  {
    v41 = *(_QWORD *)(v7 + 128)
        + (*(unsigned __int8 *)(v7 + 136) + *(_QWORD *)(v7 + 176) + (unsigned __int64)(dword_4F06BA0 - 1))
        / dword_4F06BA0;
  }
  else
  {
LABEL_60:
    v40 = *(_QWORD *)(v7 + 128);
    if ( (*(_BYTE *)(v7 + 146) & 4) != 0 && (unsigned __int8)(v29 - 9) <= 1u )
    {
      v59 = sub_730E80(v26);
      v60 = *(_QWORD *)(v26 + 168);
      if ( *(_QWORD *)(v60 + 32) >= v59 )
        v59 = *(_QWORD *)(v60 + 32);
      v41 = v40 + v59;
    }
    else
    {
      v41 = v40 + *(_QWORD *)(v26 + 128);
    }
  }
  v24 = *(_QWORD *)(v6 + 128);
  if ( *(_BYTE *)(v6 + 140) == 11 )
  {
    v23 = v67;
    if ( v41 < v24 )
      goto LABEL_65;
    return v23;
  }
  v23 = v67;
  v24 -= v41;
LABEL_65:
  v23 += v24;
LABEL_31:
  if ( v24 )
  {
LABEL_55:
    sub_918530(v24, a3, *(_QWORD *)*a1);
    *a4 += v24;
  }
  return v23;
}
