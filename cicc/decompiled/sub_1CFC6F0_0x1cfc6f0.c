// Function: sub_1CFC6F0
// Address: 0x1cfc6f0
//
__int64 __fastcall sub_1CFC6F0(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  int v10; // r12d
  unsigned int v11; // edi
  __int16 v12; // ax
  int v13; // esi
  _WORD *v14; // r8
  unsigned __int16 *v15; // rdi
  unsigned __int16 v16; // r8
  unsigned __int16 *v17; // r13
  unsigned __int16 *v18; // r10
  unsigned __int16 *v19; // rax
  int v20; // r9d
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 result; // rax
  unsigned __int16 *v24; // r11
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // r8d
  unsigned __int16 *v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  unsigned int *v31; // rdx
  unsigned int *v32; // rax
  int v33; // r14d
  __int64 v34; // r12
  unsigned int *v35; // rbx
  unsigned int v36; // ecx
  __int64 i; // r13
  unsigned int v38; // edx
  __int64 v39; // rax
  char v40; // di
  _BOOL4 v41; // r12d
  __int64 v42; // rax
  unsigned int v43; // eax
  __int64 v44; // rbx
  unsigned int v45; // edx
  __int64 v46; // rax
  char v47; // si
  _BOOL4 v48; // r13d
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 j; // rdi
  unsigned int v54; // edx
  __int64 v55; // rax
  char v56; // si
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int16 v59; // ax
  __int64 v60; // [rsp+10h] [rbp-90h]
  unsigned __int16 *v61; // [rsp+18h] [rbp-88h]
  unsigned __int16 *v62; // [rsp+20h] [rbp-80h]
  unsigned __int16 v63; // [rsp+28h] [rbp-78h]
  unsigned __int16 *v64; // [rsp+28h] [rbp-78h]
  unsigned __int16 *v65; // [rsp+28h] [rbp-78h]
  unsigned __int16 *v66; // [rsp+28h] [rbp-78h]
  unsigned __int16 *v67; // [rsp+28h] [rbp-78h]
  unsigned __int16 v68; // [rsp+30h] [rbp-70h]
  unsigned __int16 *v69; // [rsp+30h] [rbp-70h]
  unsigned __int16 *v70; // [rsp+30h] [rbp-70h]
  unsigned __int16 *v71; // [rsp+30h] [rbp-70h]
  unsigned __int16 *v72; // [rsp+30h] [rbp-70h]
  int v73; // [rsp+3Ch] [rbp-64h]
  unsigned int v74; // [rsp+3Ch] [rbp-64h]
  int v75; // [rsp+3Ch] [rbp-64h]
  int v76; // [rsp+3Ch] [rbp-64h]
  int v77; // [rsp+3Ch] [rbp-64h]
  __int64 v78; // [rsp+40h] [rbp-60h]
  const void *v82; // [rsp+60h] [rbp-40h]
  unsigned int v83; // [rsp+68h] [rbp-38h]
  unsigned __int16 v84; // [rsp+6Eh] [rbp-32h]

  if ( !a6 )
    BUG();
  v6 = a6[1];
  v7 = a6[7];
  v10 = 0;
  v11 = *(_DWORD *)(v6 + 24LL * a2 + 16);
  v12 = a2 * (v11 & 0xF);
  v13 = 0;
  v14 = (_WORD *)(v7 + 2LL * (v11 >> 4));
  v15 = v14 + 1;
  v16 = *v14 + v12;
LABEL_3:
  v17 = v15;
  while ( 1 )
  {
    v18 = v17;
    if ( !v17 )
    {
      v20 = v13;
      v21 = 0;
      goto LABEL_7;
    }
    v19 = (unsigned __int16 *)(a6[6] + 4LL * v16);
    v20 = *v19;
    v10 = v19[1];
    if ( (_WORD)v20 )
      break;
LABEL_84:
    v59 = *v17;
    v15 = 0;
    ++v17;
    if ( !v59 )
      goto LABEL_3;
    v16 += v59;
  }
  while ( 1 )
  {
    v21 = v7 + 2LL * *(unsigned int *)(v6 + 24LL * (unsigned __int16)v20 + 8);
    if ( v21 )
      break;
    if ( !(_WORD)v10 )
    {
      v13 = v20;
      goto LABEL_84;
    }
    v20 = v10;
    v10 = 0;
  }
LABEL_7:
  v22 = a5;
  v78 = a4 + 40;
LABEL_8:
  result = v22 + 16;
  v84 = v16;
  v24 = (unsigned __int16 *)v21;
  v82 = (const void *)(v22 + 16);
LABEL_9:
  if ( !v18 )
    return result;
  do
  {
    v25 = *a3;
    v26 = *(_QWORD *)(*a3 + 8LL * (unsigned __int16)v20);
    LOBYTE(v25) = v26 != 0 && v26 != a1;
    v27 = v25;
    if ( !(_BYTE)v25 )
      goto LABEL_11;
    v83 = (unsigned __int16)v20;
    if ( *(_QWORD *)(a4 + 72) )
    {
      v44 = *(_QWORD *)(a4 + 48);
      if ( v44 )
      {
        while ( 1 )
        {
          v45 = *(_DWORD *)(v44 + 32);
          v46 = *(_QWORD *)(v44 + 24);
          v47 = 0;
          if ( (unsigned __int16)v20 < v45 )
          {
            v46 = *(_QWORD *)(v44 + 16);
            v47 = v27;
          }
          if ( !v46 )
            break;
          v44 = v46;
        }
        if ( !v47 )
        {
          if ( (unsigned __int16)v20 <= v45 )
            goto LABEL_11;
          goto LABEL_48;
        }
        if ( *(_QWORD *)(a4 + 56) == v44 )
        {
LABEL_48:
          v48 = 1;
          if ( v78 == v44 )
            goto LABEL_49;
          goto LABEL_73;
        }
      }
      else
      {
        v44 = a4 + 40;
        if ( *(_QWORD *)(a4 + 56) == v78 )
        {
          v44 = a4 + 40;
          v48 = 1;
          goto LABEL_49;
        }
      }
      v65 = v24;
      v70 = v18;
      v75 = v20;
      v57 = sub_220EF80(v44);
      v20 = v75;
      v18 = v70;
      v24 = v65;
      if ( v83 <= *(_DWORD *)(v57 + 32) || !v44 )
        goto LABEL_11;
      v48 = 1;
      if ( v78 == v44 )
        goto LABEL_49;
LABEL_73:
      v48 = v83 < *(_DWORD *)(v44 + 32);
      goto LABEL_49;
    }
    v30 = *(unsigned int *)(a4 + 8);
    v31 = (unsigned int *)(*(_QWORD *)a4 + 4 * v30);
    if ( *(unsigned int **)a4 != v31 )
    {
      v32 = *(unsigned int **)a4;
      while ( (unsigned __int16)v20 != *v32 )
      {
        if ( v31 == ++v32 )
          goto LABEL_24;
      }
      if ( v31 != v32 )
        goto LABEL_11;
    }
LABEL_24:
    if ( v30 <= 3 )
    {
      if ( *(_DWORD *)(a4 + 8) >= *(_DWORD *)(a4 + 12) )
      {
        v67 = v24;
        v72 = v18;
        v77 = v20;
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 4, v27, v20);
        v24 = v67;
        v18 = v72;
        v20 = v77;
        v31 = (unsigned int *)(*(_QWORD *)a4 + 4LL * *(unsigned int *)(a4 + 8));
      }
      *v31 = v83;
      ++*(_DWORD *)(a4 + 8);
      goto LABEL_50;
    }
    v68 = v10;
    v60 = v22;
    v33 = v27;
    v34 = *(_QWORD *)(a4 + 48);
    v35 = (unsigned int *)(*(_QWORD *)a4 + 4 * v30 - 4);
    v63 = v20;
    v62 = v18;
    v61 = v24;
    if ( v34 )
    {
LABEL_26:
      v36 = *v35;
      for ( i = v34; ; i = v39 )
      {
        v38 = *(_DWORD *)(i + 32);
        v39 = *(_QWORD *)(i + 24);
        v40 = 0;
        if ( v36 < v38 )
        {
          v39 = *(_QWORD *)(i + 16);
          v40 = v33;
        }
        if ( !v39 )
          break;
      }
      if ( !v40 )
      {
        if ( v36 > v38 )
          goto LABEL_33;
        goto LABEL_36;
      }
      if ( *(_QWORD *)(a4 + 56) == i )
        goto LABEL_33;
      goto LABEL_57;
    }
    while ( 1 )
    {
      i = a4 + 40;
      if ( v78 != *(_QWORD *)(a4 + 56) )
        break;
      i = a4 + 40;
      v41 = 1;
LABEL_35:
      v42 = sub_22077B0(40);
      *(_DWORD *)(v42 + 32) = *v35;
      sub_220F040(v41, v42, i, v78);
      ++*(_QWORD *)(a4 + 72);
      v34 = *(_QWORD *)(a4 + 48);
LABEL_36:
      v43 = *(_DWORD *)(a4 + 8) - 1;
      *(_DWORD *)(a4 + 8) = v43;
      if ( !v43 )
        goto LABEL_59;
LABEL_37:
      v35 = (unsigned int *)(*(_QWORD *)a4 + 4LL * v43 - 4);
      if ( v34 )
        goto LABEL_26;
    }
    v36 = *v35;
LABEL_57:
    v74 = v36;
    v51 = sub_220EF80(i);
    v36 = v74;
    if ( v74 > *(_DWORD *)(v51 + 32) )
    {
LABEL_33:
      v41 = 1;
      if ( v78 != i )
        v41 = v36 < *(_DWORD *)(i + 32);
      goto LABEL_35;
    }
    v43 = *(_DWORD *)(a4 + 8) - 1;
    *(_DWORD *)(a4 + 8) = v43;
    if ( v43 )
      goto LABEL_37;
LABEL_59:
    v52 = v34;
    v27 = v33;
    v20 = v63;
    v18 = v62;
    v10 = v68;
    v24 = v61;
    v22 = v60;
    if ( !v52 )
    {
      v44 = a4 + 40;
      if ( v78 == *(_QWORD *)(a4 + 56) )
      {
        v48 = 1;
        v44 = a4 + 40;
        goto LABEL_49;
      }
      goto LABEL_75;
    }
    for ( j = v52; ; j = v55 )
    {
      v54 = *(_DWORD *)(j + 32);
      v55 = *(_QWORD *)(j + 24);
      v56 = 0;
      if ( v83 < v54 )
      {
        v55 = *(_QWORD *)(j + 16);
        v56 = v27;
      }
      if ( !v55 )
        break;
    }
    v44 = j;
    if ( v56 )
    {
      if ( *(_QWORD *)(a4 + 56) == j )
        goto LABEL_67;
LABEL_75:
      v58 = sub_220EF80(v44);
      v20 = v63;
      v18 = v62;
      v24 = v61;
      if ( v83 > *(_DWORD *)(v58 + 32) && v44 )
        goto LABEL_67;
      goto LABEL_50;
    }
    if ( v83 > v54 )
    {
LABEL_67:
      v48 = 1;
      if ( v78 != v44 )
        v48 = v83 < *(_DWORD *)(v44 + 32);
LABEL_49:
      v64 = v24;
      v69 = v18;
      v73 = v20;
      v49 = sub_22077B0(40);
      *(_DWORD *)(v49 + 32) = v83;
      sub_220F040(v48, v49, v44, v78);
      v20 = v73;
      v18 = v69;
      v24 = v64;
      ++*(_QWORD *)(a4 + 72);
    }
LABEL_50:
    v50 = *(unsigned int *)(v22 + 8);
    if ( (unsigned int)v50 >= *(_DWORD *)(v22 + 12) )
    {
      v66 = v24;
      v71 = v18;
      v76 = v20;
      sub_16CD150(v22, v82, 0, 4, v27, v20);
      v50 = *(unsigned int *)(v22 + 8);
      v24 = v66;
      v18 = v71;
      v20 = v76;
    }
    *(_DWORD *)(*(_QWORD *)v22 + 4 * v50) = v83;
    ++*(_DWORD *)(v22 + 8);
LABEL_11:
    result = *v24++;
    v20 += result;
    if ( (_WORD)result )
      goto LABEL_9;
    v16 = v84;
    if ( (_WORD)v10 )
    {
      v29 = (unsigned __int16)v10;
      v20 = v10;
      v10 = 0;
      v21 = a6[7] + 2LL * *(unsigned int *)(a6[1] + 24 * v29 + 8);
      goto LABEL_8;
    }
    v10 = *v18;
    v16 = v10 + v84;
    if ( !(_WORD)v10 )
    {
      v21 = 0;
      v18 = 0;
      goto LABEL_8;
    }
    ++v18;
    v84 += v10;
    v28 = (unsigned __int16 *)(a6[6] + 4LL * v16);
    v20 = *v28;
    v10 = v28[1];
    result = v22 + 16;
    v82 = (const void *)(v22 + 16);
    v24 = (unsigned __int16 *)(a6[7] + 2LL * *(unsigned int *)(a6[1] + 24LL * (unsigned __int16)v20 + 8));
  }
  while ( v18 );
  return result;
}
