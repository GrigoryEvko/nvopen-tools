// Function: sub_37B8650
// Address: 0x37b8650
//
__int64 __fastcall sub_37B8650(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  bool v8; // al
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // r15
  __int64 *v13; // rsi
  __int64 v14; // r12
  __int64 v15; // r9
  __int64 v16; // r10
  int v17; // r11d
  int v18; // r13d
  __int64 v19; // r14
  __int64 *v20; // rdi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r8
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 *v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rax
  __int64 *v29; // rdx
  unsigned int v30; // r9d
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // r8
  int v34; // eax
  unsigned int v35; // r8d
  int v36; // edx
  int v37; // r8d
  int v38; // eax
  int v39; // r9d
  __int64 v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 v44; // r8
  __int64 *v45; // rbx
  __int64 v46; // rcx
  bool v47; // zf
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // r8
  __int64 *v51; // [rsp+8h] [rbp-88h]
  int v52; // [rsp+10h] [rbp-80h]
  unsigned int v53; // [rsp+14h] [rbp-7Ch]
  int v54; // [rsp+14h] [rbp-7Ch]
  int v55; // [rsp+14h] [rbp-7Ch]
  __int64 v56; // [rsp+20h] [rbp-70h]
  __int64 *v57; // [rsp+28h] [rbp-68h]
  __int64 *v60; // [rsp+40h] [rbp-50h]
  __int64 v61; // [rsp+48h] [rbp-48h]
  __int64 *v62; // [rsp+48h] [rbp-48h]
  _QWORD v63[7]; // [rsp+58h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v57 = (__int64 *)a2;
  v56 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v60 = (__int64 *)a2;
    goto LABEL_46;
  }
  v51 = a1 + 2;
  while ( 2 )
  {
    --v56;
    v5 = &a1[result >> 4];
    v6 = a1[1];
    v7 = *v5;
    v63[0] = a4;
    v8 = sub_37B72D0((__int64)v63, v6, v7);
    v9 = *(v57 - 1);
    v61 = *a1;
    if ( !v8 )
    {
      if ( !sub_37B72D0((__int64)v63, v6, v9) )
      {
        v47 = !sub_37B72D0((__int64)v63, v7, v40);
        v41 = a1;
        if ( v47 )
        {
          *a1 = v7;
          *v5 = v61;
          v6 = *a1;
          v61 = a1[1];
          v11 = *(v57 - 1);
          goto LABEL_7;
        }
        goto LABEL_52;
      }
      v41 = a1;
LABEL_44:
      *v41 = v6;
      v41[1] = v61;
      v11 = *(v57 - 1);
      goto LABEL_7;
    }
    if ( !sub_37B72D0((__int64)v63, v7, v9) )
    {
      v47 = !sub_37B72D0((__int64)v63, v6, v10);
      v41 = a1;
      if ( !v47 )
      {
LABEL_52:
        v11 = v61;
        *v41 = v48;
        *(v57 - 1) = v61;
        v6 = *v41;
        v61 = v41[1];
        goto LABEL_7;
      }
      goto LABEL_44;
    }
    *a1 = v7;
    *v5 = v61;
    v6 = *a1;
    v11 = *(v57 - 1);
    v61 = a1[1];
LABEL_7:
    v12 = v51;
    v13 = v57;
    v14 = v61;
    v15 = *(unsigned int *)(a4 + 688);
    v16 = *(_QWORD *)(a4 + 672);
    v17 = v15;
    v18 = v15 - 1;
    while ( 1 )
    {
      v60 = v12 - 1;
      v62 = (__int64 *)(v16 + 16 * v15);
      v19 = v18 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v20 = (__int64 *)(v16 + 16 * v19);
      if ( !v17 )
        break;
      v21 = v18 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v22 = (__int64 *)(v16 + 16LL * v21);
      v23 = *v22;
      if ( *v22 != v14 )
      {
        v38 = 1;
        while ( v23 != -4096 )
        {
          v21 = v18 & (v38 + v21);
          v55 = v38 + 1;
          v22 = (__int64 *)(v16 + 16LL * v21);
          v23 = *v22;
          if ( *v22 == v14 )
            goto LABEL_11;
          v38 = v55;
        }
        v22 = (__int64 *)(v16 + 16 * v15);
      }
LABEL_11:
      v24 = *((_DWORD *)v22 + 2);
      v25 = *v20;
      v26 = (__int64 *)(v16 + 16 * v19);
      if ( *v20 == v6 )
      {
LABEL_12:
        v27 = *((_DWORD *)v26 + 2);
      }
      else
      {
        v53 = v18 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v37 = 1;
        while ( v25 != -4096 )
        {
          v49 = v37 + 1;
          v50 = v18 & (v53 + v37);
          v52 = v49;
          v53 = v50;
          v26 = (__int64 *)(v16 + 16 * v50);
          v25 = *v26;
          if ( *v26 == v6 )
            goto LABEL_12;
          v37 = v52;
        }
        v27 = *((_DWORD *)v62 + 2);
      }
      if ( v24 >= v27 )
        break;
LABEL_8:
      v14 = *v12++;
    }
    for ( --v13; v17; --v13 )
    {
      v28 = *v20;
      v29 = (__int64 *)(v16 + 16 * v19);
      if ( *v20 == v6 )
      {
LABEL_16:
        v30 = *((_DWORD *)v29 + 2);
      }
      else
      {
        v35 = v18 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v36 = 1;
        while ( v28 != -4096 )
        {
          v39 = v36 + 1;
          v35 = v18 & (v36 + v35);
          v29 = (__int64 *)(v16 + 16LL * v35);
          v28 = *v29;
          if ( *v29 == v6 )
            goto LABEL_16;
          v36 = v39;
        }
        v30 = *((_DWORD *)v62 + 2);
      }
      v31 = v18 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v32 = (__int64 *)(v16 + 16LL * v31);
      v33 = *v32;
      if ( *v32 != v11 )
      {
        v34 = 1;
        while ( v33 != -4096 )
        {
          v31 = v18 & (v34 + v31);
          v54 = v34 + 1;
          v32 = (__int64 *)(v16 + 16LL * v31);
          v33 = *v32;
          if ( *v32 == v11 )
            goto LABEL_18;
          v34 = v54;
        }
        v32 = v62;
      }
LABEL_18:
      if ( v30 >= *((_DWORD *)v32 + 2) )
        break;
      v11 = *(v13 - 1);
    }
    if ( v13 > v60 )
    {
      *(v12 - 1) = v11;
      *v13 = v14;
      v11 = *(v13 - 1);
      v15 = *(unsigned int *)(a4 + 688);
      v6 = *a1;
      v16 = *(_QWORD *)(a4 + 672);
      v17 = v15;
      v18 = v15 - 1;
      goto LABEL_8;
    }
    sub_37B8650(v60, v57, v56, a4);
    result = (char *)v60 - (char *)a1;
    if ( (char *)v60 - (char *)a1 > 128 )
    {
      if ( v56 )
      {
        v57 = v12 - 1;
        continue;
      }
LABEL_46:
      v42 = result >> 3;
      v43 = ((result >> 3) - 2) >> 1;
      sub_37B82D0((__int64)a1, v43, result >> 3, a1[v43], a4);
      do
      {
        --v43;
        sub_37B82D0((__int64)a1, v43, v42, a1[v43], v44);
      }
      while ( v43 );
      v45 = v60;
      do
      {
        v46 = *--v45;
        *v45 = *a1;
        result = sub_37B82D0((__int64)a1, 0, v45 - a1, v46, v44);
      }
      while ( (char *)v45 - (char *)a1 > 8 );
    }
    return result;
  }
}
