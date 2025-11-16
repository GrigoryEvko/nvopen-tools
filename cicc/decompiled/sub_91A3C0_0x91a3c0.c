// Function: sub_91A3C0
// Address: 0x91a3c0
//
void __fastcall sub_91A3C0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 i; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // r15
  __int64 v9; // r11
  __int64 j; // rax
  unsigned __int64 v11; // rax
  unsigned int v12; // esi
  __int64 *v13; // r9
  unsigned int v14; // r8d
  __int64 *v15; // rax
  __int64 v16; // rdi
  _DWORD *v17; // rax
  _BYTE *v18; // rsi
  unsigned __int64 v19; // rdi
  char v20; // dl
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  unsigned int v23; // eax
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // rbx
  char v34; // al
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _BYTE *v37; // rsi
  _BYTE *v38; // rsi
  _BYTE *v39; // rsi
  _BYTE *v40; // rsi
  int v41; // eax
  int v42; // eax
  int v43; // esi
  int v44; // esi
  __int64 *v45; // r8
  __int64 v46; // rdi
  int v47; // r10d
  __int64 v48; // r9
  int v49; // esi
  int v50; // esi
  __int64 *v51; // r8
  int v52; // r10d
  __int64 v53; // rdi
  __int64 v54; // [rsp+8h] [rbp-58h]
  int v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  _QWORD v59[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a3;
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(i + 160);
  if ( !v7 )
  {
    if ( !*((_BYTE *)a1 + 328) )
    {
      v8 = 0;
      goto LABEL_24;
    }
LABEL_26:
    v20 = *(_BYTE *)(i + 142);
    if ( *(_BYTE *)(i + 140) == 12 )
    {
      v21 = i;
      do
        v21 = *(_QWORD *)(v21 + 160);
      while ( *(_BYTE *)(v21 + 140) == 12 );
      v22 = *(_QWORD *)(v21 + 128);
      if ( v20 < 0 )
      {
        v23 = *(_DWORD *)(i + 136);
LABEL_31:
        v24 = v23;
        goto LABEL_32;
      }
      v24 = (unsigned int)sub_8D4AB0(i);
    }
    else
    {
      v23 = *(_DWORD *)(i + 136);
      v22 = *(_QWORD *)(i + 128);
      v24 = v23;
      if ( v20 < 0 )
        goto LABEL_31;
    }
LABEL_32:
    v25 = sub_BCCE00(**a1, 64);
    v26 = v25;
    if ( v22 > 7 && (unsigned int)v24 >= (unsigned __int64)(1LL << sub_AE5020(a1[2], v25)) )
    {
      if ( v22 > 0xF )
        v26 = sub_BCD420(v26, v22 >> 3);
      v59[0] = v26;
      v40 = *(_BYTE **)(v4 + 8);
      if ( v40 == *(_BYTE **)(v4 + 16) )
      {
        sub_9183A0(v4, v40, v59);
      }
      else
      {
        if ( v40 )
        {
          *(_QWORD *)v40 = v26;
          v40 = *(_BYTE **)(v4 + 8);
        }
        *(_QWORD *)(v4 + 8) = v40 + 8;
      }
      v22 &= 7u;
    }
    v27 = sub_BCCE00(**a1, 32);
    v28 = v27;
    if ( v22 > 3 && (unsigned int)v24 >= (unsigned __int64)(1LL << sub_AE5020(a1[2], v27)) )
    {
      if ( v22 > 7 )
        v28 = sub_BCD420(v28, v22 >> 2);
      v59[0] = v28;
      v39 = *(_BYTE **)(v4 + 8);
      if ( v39 == *(_BYTE **)(v4 + 16) )
      {
        sub_9183A0(v4, v39, v59);
      }
      else
      {
        if ( v39 )
        {
          *(_QWORD *)v39 = v28;
          v39 = *(_BYTE **)(v4 + 8);
        }
        *(_QWORD *)(v4 + 8) = v39 + 8;
      }
      v22 &= 3u;
    }
    v29 = sub_BCCE00(**a1, 16);
    v30 = v29;
    if ( v22 > 1 && (unsigned int)v24 >= (unsigned __int64)(1LL << sub_AE5020(a1[2], v29)) )
    {
      if ( v22 > 3 )
        v30 = sub_BCD420(v30, v22 >> 1);
      v59[0] = v30;
      v38 = *(_BYTE **)(v4 + 8);
      if ( v38 == *(_BYTE **)(v4 + 16) )
      {
        sub_9183A0(v4, v38, v59);
      }
      else
      {
        if ( v38 )
        {
          *(_QWORD *)v38 = v30;
          v38 = *(_BYTE **)(v4 + 8);
        }
        *(_QWORD *)(v4 + 8) = v38 + 8;
      }
      v22 &= 1u;
    }
    v31 = sub_BCCE00(**a1, 8);
    v32 = v31;
    if ( v22 && v24 >= 1LL << sub_AE5020(a1[2], v31) )
    {
      if ( v22 != 1 )
        v32 = sub_BCD420(v32, v22);
      v59[0] = v32;
      v37 = *(_BYTE **)(v4 + 8);
      if ( v37 == *(_BYTE **)(v4 + 16) )
      {
        sub_9183A0(v4, v37, v59);
      }
      else
      {
        if ( v37 )
        {
          *(_QWORD *)v37 = v32;
          v37 = *(_BYTE **)(v4 + 8);
        }
        *(_QWORD *)(v4 + 8) = v37 + 8;
      }
    }
    return;
  }
  v8 = 0;
  v9 = 0;
  v57 = (__int64)(a1 + 13);
  do
  {
    while ( (*(_BYTE *)(v7 + 146) & 8) != 0 || (*(_BYTE *)(v7 + 144) & 4) != 0 )
    {
      v7 = *(_QWORD *)(v7 + 112);
      if ( !v7 )
        goto LABEL_16;
    }
    for ( j = *(_QWORD *)(v7 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v11 = *(_QWORD *)(j + 128);
    if ( v11 > v8 )
    {
      v8 = v11;
      v9 = v7;
    }
    v12 = *((_DWORD *)a1 + 32);
    if ( v12 )
    {
      v13 = a1[14];
      a3 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
      v14 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v15 = &v13[2 * v14];
      v16 = *v15;
      if ( *v15 == v7 )
      {
LABEL_14:
        v17 = v15 + 1;
        goto LABEL_15;
      }
      v55 = 1;
      a4 = 0;
      while ( v16 != -4096 )
      {
        if ( v16 == -8192 && !a4 )
          a4 = (__int64)v15;
        v14 = (v12 - 1) & (v55 + v14);
        v15 = &v13[2 * v14];
        v16 = *v15;
        if ( *v15 == v7 )
          goto LABEL_14;
        ++v55;
      }
      if ( !a4 )
        a4 = (__int64)v15;
      v41 = *((_DWORD *)a1 + 30);
      a1[13] = (__int64 *)((char *)a1[13] + 1);
      v42 = v41 + 1;
      if ( 4 * v42 < 3 * v12 )
      {
        if ( v12 - *((_DWORD *)a1 + 31) - v42 > v12 >> 3 )
          goto LABEL_83;
        v54 = v9;
        sub_9187E0(v57, v12);
        v49 = *((_DWORD *)a1 + 32);
        if ( !v49 )
        {
LABEL_119:
          ++*((_DWORD *)a1 + 30);
          BUG();
        }
        v50 = v49 - 1;
        v51 = a1[14];
        v48 = 0;
        v9 = v54;
        v52 = 1;
        a3 = v50 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v42 = *((_DWORD *)a1 + 30) + 1;
        a4 = (__int64)&v51[2 * a3];
        v53 = *(_QWORD *)a4;
        if ( *(_QWORD *)a4 == v7 )
          goto LABEL_83;
        while ( v53 != -4096 )
        {
          if ( v53 == -8192 && !v48 )
            v48 = a4;
          a3 = v50 & (unsigned int)(v52 + a3);
          a4 = (__int64)&v51[2 * (unsigned int)a3];
          v53 = *(_QWORD *)a4;
          if ( *(_QWORD *)a4 == v7 )
            goto LABEL_83;
          ++v52;
        }
        goto LABEL_91;
      }
    }
    else
    {
      a1[13] = (__int64 *)((char *)a1[13] + 1);
    }
    v56 = v9;
    sub_9187E0(v57, 2 * v12);
    v43 = *((_DWORD *)a1 + 32);
    if ( !v43 )
      goto LABEL_119;
    v44 = v43 - 1;
    v45 = a1[14];
    v9 = v56;
    a3 = v44 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v42 = *((_DWORD *)a1 + 30) + 1;
    a4 = (__int64)&v45[2 * a3];
    v46 = *(_QWORD *)a4;
    if ( *(_QWORD *)a4 == v7 )
      goto LABEL_83;
    v47 = 1;
    v48 = 0;
    while ( v46 != -4096 )
    {
      if ( v46 == -8192 && !v48 )
        v48 = a4;
      a3 = v44 & (unsigned int)(v47 + a3);
      a4 = (__int64)&v45[2 * (unsigned int)a3];
      v46 = *(_QWORD *)a4;
      if ( *(_QWORD *)a4 == v7 )
        goto LABEL_83;
      ++v47;
    }
LABEL_91:
    if ( v48 )
      a4 = v48;
LABEL_83:
    *((_DWORD *)a1 + 30) = v42;
    if ( *(_QWORD *)a4 != -4096 )
      --*((_DWORD *)a1 + 31);
    *(_QWORD *)a4 = v7;
    v17 = (_DWORD *)(a4 + 8);
    *(_DWORD *)(a4 + 8) = 0;
LABEL_15:
    *v17 = 0;
    v7 = *(_QWORD *)(v7 + 112);
  }
  while ( v7 );
LABEL_16:
  if ( *((_BYTE *)a1 + 328) )
    goto LABEL_26;
  if ( v9 )
  {
    v59[0] = sub_91A3B0((__int64)a1, *(_QWORD *)(v9 + 120), a3, a4);
    if ( ((v8 - 4) & 0xFFFFFFFFFFFFFFFBLL) == 0 || v8 - 1 <= 1 )
    {
      v33 = sub_BCCE00(**a1, (unsigned int)(8 * v8));
      v34 = sub_AE5020(a1[2], v33);
      v35 = 1LL << v34;
      if ( *(char *)(i + 142) >= 0 && *(_BYTE *)(i + 140) == 12 )
      {
        v58 = 1LL << v34;
        LODWORD(v36) = sub_8D4AB0(i);
        v35 = v58;
        v36 = (unsigned int)v36;
      }
      else
      {
        v36 = *(unsigned int *)(i + 136);
      }
      if ( v36 >= v35 )
        v59[0] = v33;
    }
    v18 = *(_BYTE **)(v4 + 8);
    if ( v18 == *(_BYTE **)(v4 + 16) )
    {
      sub_918210(v4, v18, v59);
    }
    else
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = v59[0];
        v18 = *(_BYTE **)(v4 + 8);
      }
      *(_QWORD *)(v4 + 8) = v18 + 8;
    }
  }
LABEL_24:
  v19 = *(_QWORD *)(i + 128);
  if ( v19 > v8 )
    sub_918530(v19 - v8, v4, **a1);
}
