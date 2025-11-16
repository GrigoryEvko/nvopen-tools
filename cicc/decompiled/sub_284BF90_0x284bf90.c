// Function: sub_284BF90
// Address: 0x284bf90
//
__int64 __fastcall sub_284BF90(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // r12
  __int64 v7; // r14
  __int64 v8; // rbx
  bool v9; // al
  __int64 v10; // r15
  __int64 v11; // r11
  __int64 v12; // rbx
  __int64 *v13; // r15
  __int64 *v14; // r12
  __int64 v15; // r11
  char v16; // di
  unsigned int v17; // r10d
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // r8d
  __int64 *v21; // rax
  __int64 v22; // rcx
  int v23; // r8d
  int v24; // ecx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // eax
  int v33; // eax
  int v34; // r9d
  bool v35; // al
  __int64 *v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 *v39; // rbx
  __int64 v40; // rcx
  bool v41; // al
  bool v42; // zf
  bool v43; // al
  __int64 *v44; // [rsp+0h] [rbp-80h]
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 *v46; // [rsp+10h] [rbp-70h]
  __int64 v48; // [rsp+28h] [rbp-58h]
  int v49; // [rsp+28h] [rbp-58h]
  __int64 v50; // [rsp+30h] [rbp-50h]
  __int64 *v51; // [rsp+30h] [rbp-50h]
  __int64 v52; // [rsp+38h] [rbp-48h]
  __int64 v53[7]; // [rsp+48h] [rbp-38h] BYREF

  result = (char *)a2 - (char *)a1;
  v46 = a2;
  v45 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v51 = a2;
    goto LABEL_50;
  }
  v44 = a1 + 2;
  v52 = a4 + 16;
  while ( 2 )
  {
    --v45;
    v53[0] = a4;
    v6 = &a1[result >> 4];
    v7 = a1[1];
    v8 = *v6;
    v9 = sub_284BC70(v53, v7, *v6);
    v10 = *(v46 - 1);
    v50 = *a1;
    if ( v9 )
    {
      if ( sub_284BC70(v53, v8, v10) )
      {
        *a1 = v8;
        *v6 = v50;
        v7 = *a1;
        v11 = a1[1];
        v12 = *(v46 - 1);
        goto LABEL_7;
      }
      v41 = sub_284BC70(v53, v7, v10);
      v11 = v50;
      v42 = !v41;
      v36 = a1;
      if ( !v42 )
        goto LABEL_56;
LABEL_48:
      *v36 = v7;
      v36[1] = v11;
      v12 = *(v46 - 1);
      goto LABEL_7;
    }
    v35 = sub_284BC70(v53, v7, v10);
    v11 = v50;
    if ( v35 )
    {
      v36 = a1;
      goto LABEL_48;
    }
    v43 = sub_284BC70(v53, v8, v10);
    v11 = v50;
    v42 = !v43;
    v36 = a1;
    if ( !v42 )
    {
LABEL_56:
      *v36 = v10;
      v12 = v11;
      *(v46 - 1) = v11;
      v7 = *v36;
      v11 = v36[1];
      goto LABEL_7;
    }
    *a1 = v8;
    *v6 = v50;
    v7 = *a1;
    v11 = a1[1];
    v12 = *(v46 - 1);
LABEL_7:
    v13 = v44;
    v14 = v46;
    v53[0] = a4;
    while ( 1 )
    {
      v51 = v13 - 1;
      v48 = v11;
      if ( !sub_284BC70(v53, v11, v7) )
        break;
LABEL_24:
      v11 = *v13++;
    }
    v15 = v48;
    --v14;
    v16 = *(_BYTE *)(a4 + 8) & 1;
    v17 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
    while ( 1 )
    {
      if ( v16 )
      {
        v18 = v52;
        v19 = 15;
      }
      else
      {
        v28 = *(unsigned int *)(a4 + 24);
        v18 = *(_QWORD *)(a4 + 16);
        if ( !(_DWORD)v28 )
          goto LABEL_26;
        v19 = v28 - 1;
      }
      v20 = v17 & v19;
      v21 = (__int64 *)(v18 + 16LL * (v17 & v19));
      v22 = *v21;
      if ( *v21 == v7 )
        goto LABEL_12;
      v33 = 1;
      while ( v22 != -4096 )
      {
        v34 = v33 + 1;
        v20 = v19 & (v33 + v20);
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == v7 )
          goto LABEL_12;
        v33 = v34;
      }
      if ( v16 )
      {
        v30 = 256;
        goto LABEL_27;
      }
      v28 = *(unsigned int *)(a4 + 24);
LABEL_26:
      v30 = 16 * v28;
LABEL_27:
      v21 = (__int64 *)(v18 + v30);
LABEL_12:
      v23 = *((_DWORD *)v21 + 2);
      if ( v16 )
      {
        v24 = 15;
      }
      else
      {
        v29 = *(unsigned int *)(a4 + 24);
        if ( !(_DWORD)v29 )
          goto LABEL_29;
        v24 = v29 - 1;
      }
      v25 = v24 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v26 = (__int64 *)(v18 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == v12 )
        goto LABEL_15;
      v32 = 1;
      while ( v27 != -4096 )
      {
        v25 = v24 & (v32 + v25);
        v49 = v32 + 1;
        v26 = (__int64 *)(v18 + 16LL * v25);
        v27 = *v26;
        if ( *v26 == v12 )
          goto LABEL_15;
        v32 = v49;
      }
      if ( v16 )
      {
        v31 = 256;
        goto LABEL_30;
      }
      v29 = *(unsigned int *)(a4 + 24);
LABEL_29:
      v31 = 16 * v29;
LABEL_30:
      v26 = (__int64 *)(v18 + v31);
LABEL_15:
      if ( v23 >= *((_DWORD *)v26 + 2) )
        break;
      v12 = *--v14;
    }
    if ( v14 > v51 )
    {
      *(v13 - 1) = v12;
      *v14 = v15;
      v12 = *(v14 - 1);
      v7 = *a1;
      goto LABEL_24;
    }
    sub_284BF90(v51, v46, v45, a4);
    result = (char *)v51 - (char *)a1;
    if ( (char *)v51 - (char *)a1 > 128 )
    {
      if ( v45 )
      {
        v46 = v13 - 1;
        continue;
      }
LABEL_50:
      v37 = result >> 3;
      v38 = ((result >> 3) - 2) >> 1;
      sub_284BDE0((__int64)a1, v38, result >> 3, a1[v38], a4);
      do
      {
        --v38;
        sub_284BDE0((__int64)a1, v38, v37, a1[v38], a4);
      }
      while ( v38 );
      v39 = v51;
      do
      {
        v40 = *--v39;
        *v39 = *a1;
        result = sub_284BDE0((__int64)a1, 0, v39 - a1, v40, a4);
      }
      while ( (char *)v39 - (char *)a1 > 8 );
    }
    return result;
  }
}
