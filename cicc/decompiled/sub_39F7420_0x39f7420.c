// Function: sub_39F7420
// Address: 0x39f7420
//
__int64 __fastcall sub_39F7420(_QWORD *a1, char *a2)
{
  __int64 v3; // rdx
  __int64 v5; // rax
  unsigned int *v6; // r13
  unsigned int *v7; // r14
  char *v8; // rbx
  char *v9; // rax
  unsigned __int8 v10; // di
  __int64 v11; // r8
  int v12; // ecx
  char v13; // si
  unsigned __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 v16; // r8
  char *v17; // r9
  char v18; // si
  unsigned __int64 v19; // rdx
  int v20; // ecx
  __int64 v21; // r8
  char v22; // si
  unsigned __int64 v23; // rdx
  char v24; // dl
  char *v25; // r15
  char *v26; // rbx
  unsigned __int64 *v27; // rcx
  char v28; // dl
  char v29; // al
  unsigned __int8 v30; // al
  __int64 v31; // rdx
  char *v32; // rdx
  char v33; // r8
  __int64 v34; // rdi
  int v35; // ecx
  char v36; // si
  unsigned __int64 v37; // rax
  char *v38; // r14
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // ecx
  char v44; // si
  unsigned __int64 v45; // rdx
  char v46; // dl
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int64 *v50; // [rsp+8h] [rbp-50h]
  unsigned __int64 v51[8]; // [rsp+18h] [rbp-40h] BYREF

  memset(a2, 0, 0x180u);
  a1[26] = 0;
  a1[20] = 0;
  v3 = a1[19];
  if ( !v3 )
    return 5;
  v5 = sub_39FA9E0(v3 + (a1[24] >> 63) - 1, a1 + 21);
  v6 = (unsigned int *)v5;
  if ( !v5 )
  {
    v47 = a1[19];
    if ( *(_BYTE *)v47 == 72 && *(_QWORD *)(v47 + 1) == 0x50F0000000FC0C7LL )
    {
      v48 = a1[18];
      v49 = *(_QWORD *)(v48 + 160);
      *((_DWORD *)a2 + 80) = 1;
      *((_QWORD *)a2 + 38) = 7;
      *((_DWORD *)a2 + 2) = 1;
      *(_QWORD *)a2 = v48 + 144 - v49;
      *((_QWORD *)a2 + 37) = v49 - v48;
      *((_QWORD *)a2 + 2) = v48 + 136 - v49;
      *((_QWORD *)a2 + 16) = v48 + 40 - v49;
      *((_QWORD *)a2 + 4) = v48 + 152 - v49;
      *((_QWORD *)a2 + 18) = v48 + 48 - v49;
      *((_QWORD *)a2 + 6) = v48 + 128 - v49;
      *((_QWORD *)a2 + 20) = v48 + 56 - v49;
      *((_QWORD *)a2 + 8) = v48 + 112 - v49;
      *((_QWORD *)a2 + 22) = v48 + 64 - v49;
      *((_QWORD *)a2 + 10) = v48 + 104 - v49;
      *((_QWORD *)a2 + 24) = v48 + 72 - v49;
      *((_DWORD *)a2 + 6) = 1;
      *((_QWORD *)a2 + 26) = v48 + 80 - v49;
      *((_DWORD *)a2 + 10) = 1;
      *((_DWORD *)a2 + 14) = 1;
      *((_DWORD *)a2 + 18) = 1;
      *((_DWORD *)a2 + 22) = 1;
      *((_DWORD *)a2 + 26) = 1;
      *((_QWORD *)a2 + 12) = v48 + 120 - v49;
      *((_DWORD *)a2 + 34) = 1;
      *((_DWORD *)a2 + 38) = 1;
      *((_DWORD *)a2 + 42) = 1;
      *((_DWORD *)a2 + 46) = 1;
      *((_DWORD *)a2 + 50) = 1;
      *((_DWORD *)a2 + 54) = 1;
      *((_DWORD *)a2 + 58) = 1;
      *((_QWORD *)a2 + 28) = v48 + 88 - v49;
      *((_DWORD *)a2 + 62) = 1;
      *((_QWORD *)a2 + 32) = v48 + 168 - v49;
      *((_QWORD *)a2 + 30) = v48 + 96 - v49;
      *((_DWORD *)a2 + 66) = 1;
      *((_QWORD *)a2 + 45) = 16;
      a2[371] = 1;
      return 0;
    }
    return 5;
  }
  *((_QWORD *)a2 + 41) = a1[23];
  v7 = (unsigned int *)(v5 + 4 - *(int *)(v5 + 4));
  v8 = (char *)v7 + 9;
  v9 = (char *)v7 + strlen((const char *)v7 + 9) + 10;
  if ( *((_BYTE *)v7 + 9) == 101 && *((_BYTE *)v7 + 10) == 104 )
  {
    v40 = *(_QWORD *)v9;
    v8 = (char *)v7 + 11;
    v9 += 8;
    *((_QWORD *)a2 + 47) = v40;
  }
  v10 = *((_BYTE *)v7 + 8);
  if ( v10 > 3u )
  {
    if ( *v9 != 8 || v9[1] )
      return 3;
    v9 += 2;
  }
  v11 = 0;
  v12 = 0;
  do
  {
    v13 = *v9++;
    v14 = (unsigned __int64)(v13 & 0x7F) << v12;
    v12 += 7;
    v11 |= v14;
  }
  while ( v13 < 0 );
  *((_QWORD *)a2 + 44) = v11;
  v15 = 0;
  v16 = 0;
  do
  {
    v17 = v9;
    v18 = *v9++;
    v19 = (unsigned __int64)(v18 & 0x7F) << v15;
    v15 += 7;
    v16 |= v19;
  }
  while ( v18 < 0 );
  if ( v15 <= 0x3F && (v18 & 0x40) != 0 )
    v16 |= -1LL << v15;
  *((_QWORD *)a2 + 43) = v16;
  v20 = 0;
  v21 = 0;
  if ( v10 == 1 )
  {
    v41 = (unsigned __int8)*v9;
    a2[369] = -1;
    *((_QWORD *)a2 + 45) = v41;
    v24 = *v8;
    v9 = v17 + 2;
    if ( *v8 != 122 )
      goto LABEL_15;
  }
  else
  {
    do
    {
      v22 = *v9++;
      v23 = (unsigned __int64)(v22 & 0x7F) << v20;
      v20 += 7;
      v21 |= v23;
    }
    while ( v22 < 0 );
    *((_QWORD *)a2 + 45) = v21;
    a2[369] = -1;
    v24 = *v8;
    if ( *v8 != 122 )
    {
LABEL_15:
      v25 = 0;
      if ( !v24 )
      {
        v25 = v9;
        goto LABEL_25;
      }
      goto LABEL_16;
    }
  }
  v42 = 0;
  v43 = 0;
  do
  {
    v44 = *v9++;
    v45 = (unsigned __int64)(v44 & 0x7F) << v43;
    v43 += 7;
    v42 |= v45;
  }
  while ( v44 < 0 );
  a2[370] = 1;
  v24 = v8[1];
  v25 = &v9[v42];
  if ( !v24 )
    goto LABEL_25;
  ++v8;
LABEL_16:
  v26 = v8 + 1;
  v27 = v51;
  do
  {
    while ( v24 == 76 )
    {
      v28 = *v9;
      ++v26;
      ++v9;
      a2[369] = v28;
      v24 = *(v26 - 1);
      if ( !v24 )
        goto LABEL_24;
    }
    switch ( v24 )
    {
      case 'R':
        v46 = *v9++;
        a2[368] = v46;
        break;
      case 'P':
        v50 = v27;
        v9 = sub_39F5E90(a1, *v9, v9 + 1, v27);
        v27 = v50;
        *((_QWORD *)a2 + 42) = v51[0];
        break;
      case 'S':
        a2[371] = 1;
        break;
      case 'B':
        break;
      default:
        goto LABEL_53;
    }
    v24 = *v26++;
  }
  while ( v24 );
LABEL_24:
  if ( !v25 )
  {
    v25 = v9;
LABEL_53:
    if ( !v25 )
      return 3;
  }
LABEL_25:
  sub_39F6BB0(v25, (unsigned __int64)v7 + *v7 + 4, a1, (__int64)a2);
  v29 = a2[368];
  if ( v29 == -1 )
  {
    v31 = 2;
    goto LABEL_29;
  }
  v30 = v29 & 7;
  if ( v30 == 2 )
  {
    v31 = 3;
    goto LABEL_29;
  }
  if ( v30 <= 2u )
  {
    if ( !v30 )
      goto LABEL_46;
    goto LABEL_64;
  }
  v31 = 4;
  if ( v30 != 3 )
  {
    if ( v30 == 4 )
    {
LABEL_46:
      v31 = 6;
      goto LABEL_29;
    }
LABEL_64:
    abort();
  }
LABEL_29:
  v32 = (char *)&v6[v31];
  v33 = a2[369];
  if ( a2[370] )
  {
    v34 = 0;
    v35 = 0;
    do
    {
      v36 = *v32++;
      v37 = (unsigned __int64)(v36 & 0x7F) << v35;
      v35 += 7;
      v34 |= v37;
    }
    while ( v36 < 0 );
    v38 = &v32[v34];
    if ( v33 != -1 )
    {
      sub_39F5E90(a1, v33, v32, v51);
      a1[20] = v51[0];
    }
  }
  else
  {
    v38 = v32;
    if ( v33 != -1 )
    {
      v38 = sub_39F5E90(a1, v33, v32, v51);
      a1[20] = v51[0];
    }
  }
  sub_39F6BB0(v38, (unsigned __int64)v6 + *v6 + 4, a1, (__int64)a2);
  return 0;
}
