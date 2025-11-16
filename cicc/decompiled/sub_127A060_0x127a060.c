// Function: sub_127A060
// Address: 0x127a060
//
void __fastcall sub_127A060(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 i; // r12
  __int64 v6; // rbx
  unsigned __int64 v7; // r15
  __int64 v8; // r11
  __int64 v9; // r10
  __int64 j; // rax
  unsigned __int64 v11; // rax
  unsigned int v12; // esi
  __int64 *v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // rsi
  unsigned __int64 v18; // rdi
  char v19; // dl
  __int64 v20; // rax
  unsigned __int64 v21; // r15
  unsigned int v22; // eax
  unsigned int v23; // ebx
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rbx
  unsigned int v33; // eax
  unsigned int v34; // edx
  unsigned int v35; // eax
  _BYTE *v36; // rsi
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // r12
  __int64 v40; // r12
  __int64 v41; // r12
  _BYTE *v42; // rsi
  _BYTE *v43; // rsi
  _BYTE *v44; // rsi
  int v45; // esi
  int v46; // esi
  __int64 *v47; // r8
  unsigned int v48; // edx
  __int64 v49; // rdi
  __int64 *v50; // r9
  int v51; // esi
  int v52; // esi
  __int64 *v53; // r8
  unsigned int v54; // edx
  __int64 v55; // rdi
  __int64 v56; // [rsp+8h] [rbp-58h]
  int v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+10h] [rbp-50h]
  unsigned int v59; // [rsp+10h] [rbp-50h]
  __int64 *v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  int v62; // [rsp+18h] [rbp-48h]
  __int64 v63; // [rsp+18h] [rbp-48h]
  int v64; // [rsp+18h] [rbp-48h]
  unsigned int v65; // [rsp+18h] [rbp-48h]
  _QWORD v66[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(_QWORD *)(i + 160);
  if ( !v6 )
  {
    if ( !*((_BYTE *)a1 + 344) )
    {
      v7 = 0;
      goto LABEL_23;
    }
LABEL_25:
    v19 = *(_BYTE *)(i + 142);
    if ( *(_BYTE *)(i + 140) == 12 )
    {
      v20 = i;
      do
        v20 = *(_QWORD *)(v20 + 160);
      while ( *(_BYTE *)(v20 + 140) == 12 );
      v21 = *(_QWORD *)(v20 + 128);
      if ( v19 < 0 )
      {
        v22 = *(_DWORD *)(i + 136);
LABEL_30:
        v23 = v22;
        goto LABEL_31;
      }
      v23 = sub_8D4AB0(i);
    }
    else
    {
      v22 = *(_DWORD *)(i + 136);
      v21 = *(_QWORD *)(i + 128);
      v23 = v22;
      if ( v19 < 0 )
        goto LABEL_30;
    }
LABEL_31:
    v24 = sub_1644900(**a1, 64);
    v25 = v24;
    if ( v21 > 7 )
    {
      if ( v23 < (unsigned int)sub_15A9FE0(a1[2], v24) )
      {
        v39 = sub_1644900(**a1, 32);
        if ( v23 >= (unsigned int)sub_15A9FE0(a1[2], v39) )
        {
          v27 = sub_1645D80(v39, v21 >> 2);
          goto LABEL_71;
        }
LABEL_63:
        v40 = sub_1644900(**a1, 16);
        if ( v23 >= (unsigned int)sub_15A9FE0(a1[2], v40) )
        {
          v29 = sub_1645D80(v40, v21 >> 1);
          goto LABEL_79;
        }
LABEL_64:
        v41 = sub_1644900(**a1, 8);
        if ( v23 < (unsigned int)sub_15A9FE0(a1[2], v41) )
          return;
        v31 = sub_1645D80(v41, v21);
LABEL_66:
        v66[0] = v31;
        v42 = *(_BYTE **)(a3 + 8);
        if ( v42 == *(_BYTE **)(a3 + 16) )
        {
          sub_1278040(a3, v42, v66);
        }
        else
        {
          if ( v42 )
          {
            *(_QWORD *)v42 = v31;
            v42 = *(_BYTE **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v42 + 8;
        }
        return;
      }
      if ( v21 > 0xF )
        v25 = sub_1645D80(v25, v21 >> 3);
      v66[0] = v25;
      v36 = *(_BYTE **)(a3 + 8);
      if ( v36 == *(_BYTE **)(a3 + 16) )
      {
        sub_1278040(a3, v36, v66);
      }
      else
      {
        if ( v36 )
        {
          *(_QWORD *)v36 = v25;
          v36 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v36 + 8;
      }
      v21 &= 7u;
    }
    v26 = sub_1644900(**a1, 32);
    v27 = v26;
    if ( v21 <= 3 )
      goto LABEL_33;
    if ( v23 >= (unsigned int)sub_15A9FE0(a1[2], v26) )
    {
LABEL_71:
      v66[0] = v27;
      v43 = *(_BYTE **)(a3 + 8);
      if ( v43 == *(_BYTE **)(a3 + 16) )
      {
        sub_1278040(a3, v43, v66);
      }
      else
      {
        if ( v43 )
        {
          *(_QWORD *)v43 = v27;
          v43 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v43 + 8;
      }
      v21 &= 3u;
LABEL_33:
      v28 = sub_1644900(**a1, 16);
      v29 = v28;
      if ( v21 <= 1 )
        goto LABEL_34;
      if ( v23 >= (unsigned int)sub_15A9FE0(a1[2], v28) )
      {
LABEL_79:
        v66[0] = v29;
        v44 = *(_BYTE **)(a3 + 8);
        if ( v44 == *(_BYTE **)(a3 + 16) )
        {
          sub_1278040(a3, v44, v66);
        }
        else
        {
          if ( v44 )
          {
            *(_QWORD *)v44 = v29;
            v44 = *(_BYTE **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v44 + 8;
        }
        v21 &= 1u;
LABEL_34:
        v30 = sub_1644900(**a1, 8);
        v31 = v30;
        if ( !v21 || v23 < (unsigned int)sub_15A9FE0(a1[2], v30) )
          return;
        goto LABEL_66;
      }
      goto LABEL_64;
    }
    goto LABEL_63;
  }
  v7 = 0;
  v8 = 0;
  v9 = (__int64)(a1 + 13);
  do
  {
    while ( (*(_BYTE *)(v6 + 146) & 8) != 0 || (*(_BYTE *)(v6 + 144) & 4) != 0 )
    {
      v6 = *(_QWORD *)(v6 + 112);
      if ( !v6 )
        goto LABEL_15;
    }
    for ( j = *(_QWORD *)(v6 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v11 = *(_QWORD *)(j + 128);
    if ( v11 > v7 )
    {
      v7 = v11;
      v8 = v6;
    }
    v12 = *((_DWORD *)a1 + 32);
    if ( v12 )
    {
      v13 = a1[14];
      v14 = (v12 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v15 = &v13[2 * v14];
      v16 = *v15;
      if ( v6 == *v15 )
        goto LABEL_14;
      v57 = 1;
      v60 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 )
        {
          if ( v60 )
            v15 = v60;
          v60 = v15;
        }
        v14 = (v12 - 1) & (v57 + v14);
        v15 = &v13[2 * v14];
        v16 = *v15;
        if ( *v15 == v6 )
          goto LABEL_14;
        ++v57;
      }
      if ( v60 )
        v15 = v60;
      v37 = *((_DWORD *)a1 + 30);
      a1[13] = (__int64 *)((char *)a1[13] + 1);
      v38 = v37 + 1;
      if ( 4 * v38 < 3 * v12 )
      {
        if ( v12 - *((_DWORD *)a1 + 31) - v38 > v12 >> 3 )
          goto LABEL_59;
        v56 = v8;
        v59 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
        v63 = v9;
        sub_1278480(v9, v12);
        v51 = *((_DWORD *)a1 + 32);
        if ( !v51 )
        {
LABEL_121:
          ++*((_DWORD *)a1 + 30);
          BUG();
        }
        v52 = v51 - 1;
        v53 = a1[14];
        v9 = v63;
        v54 = v52 & v59;
        v8 = v56;
        v38 = *((_DWORD *)a1 + 30) + 1;
        v15 = &v53[2 * (v52 & v59)];
        v55 = *v15;
        if ( v6 == *v15 )
          goto LABEL_59;
        v64 = 1;
        v50 = 0;
        while ( v55 != -8 )
        {
          if ( v55 == -16 && !v50 )
            v50 = v15;
          v54 = v52 & (v64 + v54);
          v15 = &v53[2 * v54];
          v55 = *v15;
          if ( v6 == *v15 )
            goto LABEL_59;
          ++v64;
        }
        goto LABEL_89;
      }
    }
    else
    {
      a1[13] = (__int64 *)((char *)a1[13] + 1);
    }
    v58 = v8;
    v61 = v9;
    sub_1278480(v9, 2 * v12);
    v45 = *((_DWORD *)a1 + 32);
    if ( !v45 )
      goto LABEL_121;
    v46 = v45 - 1;
    v47 = a1[14];
    v9 = v61;
    v8 = v58;
    v48 = v46 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v38 = *((_DWORD *)a1 + 30) + 1;
    v15 = &v47[2 * v48];
    v49 = *v15;
    if ( *v15 == v6 )
      goto LABEL_59;
    v62 = 1;
    v50 = 0;
    while ( v49 != -8 )
    {
      if ( v49 == -16 && !v50 )
        v50 = v15;
      v48 = v46 & (v62 + v48);
      v15 = &v47[2 * v48];
      v49 = *v15;
      if ( v6 == *v15 )
        goto LABEL_59;
      ++v62;
    }
LABEL_89:
    if ( v50 )
      v15 = v50;
LABEL_59:
    *((_DWORD *)a1 + 30) = v38;
    if ( *v15 != -8 )
      --*((_DWORD *)a1 + 31);
    *v15 = v6;
    *((_DWORD *)v15 + 2) = 0;
LABEL_14:
    *((_DWORD *)v15 + 2) = 0;
    v6 = *(_QWORD *)(v6 + 112);
  }
  while ( v6 );
LABEL_15:
  if ( *((_BYTE *)a1 + 344) )
    goto LABEL_25;
  if ( v8 )
  {
    v66[0] = sub_127A050((__int64)a1, *(_QWORD *)(v8 + 120));
    if ( ((v7 - 4) & 0xFFFFFFFFFFFFFFFBLL) == 0 || v7 - 1 <= 1 )
    {
      v32 = sub_1644900(**a1, (unsigned int)(8 * v7));
      v33 = sub_15A9FE0(a1[2], v32);
      v34 = v33;
      if ( *(char *)(i + 142) >= 0 && *(_BYTE *)(i + 140) == 12 )
      {
        v65 = v33;
        v35 = sub_8D4AB0(i);
        v34 = v65;
      }
      else
      {
        v35 = *(_DWORD *)(i + 136);
      }
      if ( v34 <= v35 )
        v66[0] = v32;
    }
    v17 = *(_BYTE **)(a3 + 8);
    if ( v17 == *(_BYTE **)(a3 + 16) )
    {
      sub_1277EB0(a3, v17, v66);
    }
    else
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = v66[0];
        v17 = *(_BYTE **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v17 + 8;
    }
  }
LABEL_23:
  v18 = *(_QWORD *)(i + 128);
  if ( v18 > v7 )
    sub_12781D0(v18 - v7, a3, **a1);
}
