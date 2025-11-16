// Function: sub_2C9D230
// Address: 0x2c9d230
//
__int64 __fastcall sub_2C9D230(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned __int64 v8; // rsi
  __int64 result; // rax
  _QWORD *v10; // rax
  __int64 v11; // r15
  __int64 v12; // r8
  _QWORD *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // r15d
  __int64 v20; // r11
  unsigned int v21; // ecx
  __int64 v22; // rdx
  int v23; // r10d
  unsigned __int64 *v24; // rdi
  unsigned int v25; // esi
  __int64 v26; // rax
  unsigned __int64 v27; // r8
  unsigned int v28; // r13d
  int v29; // r10d
  unsigned __int64 *v30; // rdi
  unsigned int v31; // esi
  __int64 v32; // rax
  unsigned __int64 v33; // r8
  unsigned int v34; // edx
  __int64 v35; // rbx
  __int64 v36; // r14
  __int64 v37; // r9
  int v38; // r11d
  unsigned __int64 *v39; // rdi
  unsigned int v40; // ecx
  __int64 v41; // rdx
  unsigned __int64 v42; // r8
  int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // ecx
  int v50; // eax
  int v51; // edx
  unsigned int v52; // eax
  int v53; // eax
  int v54; // eax
  int v55; // esi
  __int64 v56; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+8h] [rbp-68h]
  __int64 v58; // [rsp+8h] [rbp-68h]
  unsigned __int64 v59; // [rsp+18h] [rbp-58h] BYREF
  __int64 v60; // [rsp+20h] [rbp-50h] BYREF
  __int64 v61; // [rsp+28h] [rbp-48h] BYREF
  __int64 v62; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v63[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a4;
  v5 = a3;
  v8 = *(_QWORD *)(a3 + 40);
  v59 = v8;
  if ( v8 != *(_QWORD *)(a4 + 40) )
    return sub_B19DB0(a2, a3, a4);
  v10 = *(_QWORD **)(a1 + 280);
  v11 = a1 + 272;
  v12 = a1 + 272;
  if ( !v10 )
    goto LABEL_46;
  v13 = *(_QWORD **)(a1 + 280);
  do
  {
    while ( 1 )
    {
      v14 = v13[2];
      v15 = v13[3];
      if ( v13[4] >= v8 )
        break;
      v13 = (_QWORD *)v13[3];
      if ( !v15 )
        goto LABEL_9;
    }
    v12 = (__int64)v13;
    v13 = (_QWORD *)v13[2];
  }
  while ( v14 );
LABEL_9:
  if ( v11 != v12 )
  {
    if ( *(_QWORD *)(v12 + 32) <= v8 )
      goto LABEL_11;
    v12 = a1 + 272;
  }
  do
  {
    if ( v10[4] < v8 )
    {
      v10 = (_QWORD *)v10[3];
    }
    else
    {
      v12 = (__int64)v10;
      v10 = (_QWORD *)v10[2];
    }
  }
  while ( v10 );
  if ( v11 == v12 || *(_QWORD *)(v12 + 32) > v8 )
  {
LABEL_46:
    v63[0] = &v59;
    v47 = sub_2C96770((_QWORD *)(a1 + 264), v12, v63);
    v8 = v59;
    v12 = v47;
  }
  sub_2C9CD90(v12 + 40, v8);
  v10 = *(_QWORD **)(a1 + 280);
  if ( !v10 )
  {
    v16 = a1 + 272;
    goto LABEL_49;
  }
LABEL_11:
  v16 = a1 + 272;
  do
  {
    while ( 1 )
    {
      v17 = v10[2];
      v18 = v10[3];
      if ( v10[4] >= v59 )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v18 )
        goto LABEL_15;
    }
    v16 = (__int64)v10;
    v10 = (_QWORD *)v10[2];
  }
  while ( v17 );
LABEL_15:
  if ( v11 == v16 || *(_QWORD *)(v16 + 32) > v59 )
  {
LABEL_49:
    v63[0] = &v59;
    v48 = sub_2C96770((_QWORD *)(a1 + 264), v16, v63);
    v60 = v5;
    v61 = v4;
    v16 = v48;
    if ( v5 == v4 )
      return 0;
    goto LABEL_18;
  }
  v60 = v5;
  v61 = v4;
  if ( v5 == v4 )
    return 0;
LABEL_18:
  v19 = *(_DWORD *)(v16 + 64);
  v20 = v16 + 40;
  if ( !v19 )
  {
    v63[0] = 0;
    ++*(_QWORD *)(v16 + 40);
LABEL_95:
    v55 = 2 * v19;
LABEL_96:
    sub_9BAAD0(v16 + 40, v55);
    sub_2ABDB00(v16 + 40, &v60, v63);
    v5 = v60;
    v24 = v63[0];
    v20 = v16 + 40;
    v51 = *(_DWORD *)(v16 + 56) + 1;
    goto LABEL_73;
  }
  v21 = v19 - 1;
  v22 = *(_QWORD *)(v16 + 48);
  v23 = 1;
  v24 = 0;
  v25 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v26 = v22 + 16LL * v25;
  v27 = *(_QWORD *)v26;
  if ( v5 == *(_QWORD *)v26 )
  {
LABEL_20:
    v28 = *(_DWORD *)(v26 + 8);
    goto LABEL_21;
  }
  while ( v27 != -4096 )
  {
    if ( !v24 && v27 == -8192 )
      v24 = (unsigned __int64 *)v26;
    v25 = v21 & (v23 + v25);
    v26 = v22 + 16LL * v25;
    v27 = *(_QWORD *)v26;
    if ( v5 == *(_QWORD *)v26 )
      goto LABEL_20;
    ++v23;
  }
  if ( !v24 )
    v24 = (unsigned __int64 *)v26;
  v63[0] = v24;
  v50 = *(_DWORD *)(v16 + 56);
  ++*(_QWORD *)(v16 + 40);
  v51 = v50 + 1;
  if ( 4 * (v50 + 1) >= 3 * v19 )
    goto LABEL_95;
  if ( v19 - *(_DWORD *)(v16 + 60) - v51 <= v19 >> 3 )
  {
    v55 = v19;
    goto LABEL_96;
  }
LABEL_73:
  *(_DWORD *)(v16 + 56) = v51;
  if ( *v24 != -4096 )
    --*(_DWORD *)(v16 + 60);
  *v24 = v5;
  *((_DWORD *)v24 + 2) = 0;
  v19 = *(_DWORD *)(v16 + 64);
  if ( !v19 )
  {
    v63[0] = 0;
    v52 = 0;
    ++*(_QWORD *)(v16 + 40);
    goto LABEL_77;
  }
  v22 = *(_QWORD *)(v16 + 48);
  v4 = v61;
  v21 = v19 - 1;
  v28 = 0;
LABEL_21:
  v29 = 1;
  v30 = 0;
  v31 = v21 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v32 = v22 + 16LL * v31;
  v33 = *(_QWORD *)v32;
  if ( v4 != *(_QWORD *)v32 )
  {
    while ( v33 != -4096 )
    {
      if ( !v30 && v33 == -8192 )
        v30 = (unsigned __int64 *)v32;
      v31 = v21 & (v29 + v31);
      v32 = v22 + 16LL * v31;
      v33 = *(_QWORD *)v32;
      if ( v4 == *(_QWORD *)v32 )
        goto LABEL_22;
      ++v29;
    }
    if ( !v30 )
      v30 = (unsigned __int64 *)v32;
    v63[0] = v30;
    v54 = *(_DWORD *)(v16 + 56);
    ++*(_QWORD *)(v16 + 40);
    v53 = v54 + 1;
    if ( 4 * v53 < 3 * v19 )
    {
      if ( v19 - (v53 + *(_DWORD *)(v16 + 60)) <= v19 >> 3 )
      {
        v58 = v20;
        sub_9BAAD0(v20, v19);
        sub_2ABDB00(v58, &v61, v63);
        v4 = v61;
        v30 = v63[0];
        v20 = v58;
        v53 = *(_DWORD *)(v16 + 56) + 1;
      }
      goto LABEL_78;
    }
    v52 = v19;
    v19 = v28;
LABEL_77:
    v57 = v20;
    v28 = v19;
    sub_9BAAD0(v20, 2 * v52);
    sub_2ABDB00(v57, &v61, v63);
    v4 = v61;
    v30 = v63[0];
    v20 = v57;
    v53 = *(_DWORD *)(v16 + 56) + 1;
LABEL_78:
    *(_DWORD *)(v16 + 56) = v53;
    if ( *v30 != -4096 )
      --*(_DWORD *)(v16 + 60);
    *v30 = v4;
    v34 = 0;
    *((_DWORD *)v30 + 2) = 0;
LABEL_23:
    if ( v34 >= v28 )
    {
      v35 = v60 + 24;
      v36 = *(_QWORD *)(v60 + 40) + 48LL;
      if ( v60 + 24 != v36 )
      {
        v56 = v20;
        while ( 1 )
        {
          v44 = v35 - 24;
          if ( !v35 )
            v44 = 0;
          v62 = v44;
          if ( v44 == v61 )
            return 1;
          v45 = *(_DWORD *)(v16 + 64);
          if ( !v45 )
            break;
          v37 = *(_QWORD *)(v16 + 48);
          v38 = 1;
          v39 = 0;
          v40 = (v45 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v41 = v37 + 16LL * v40;
          v42 = *(_QWORD *)v41;
          if ( v44 != *(_QWORD *)v41 )
          {
            while ( v42 != -4096 )
            {
              if ( !v39 && v42 == -8192 )
                v39 = (unsigned __int64 *)v41;
              v40 = (v45 - 1) & (v38 + v40);
              v41 = v37 + 16LL * v40;
              v42 = *(_QWORD *)v41;
              if ( v44 == *(_QWORD *)v41 )
                goto LABEL_27;
              ++v38;
            }
            if ( !v39 )
              v39 = (unsigned __int64 *)v41;
            v63[0] = v39;
            v49 = *(_DWORD *)(v16 + 56);
            ++*(_QWORD *)(v16 + 40);
            v46 = v49 + 1;
            if ( 4 * v46 < 3 * v45 )
            {
              if ( v45 - *(_DWORD *)(v16 + 60) - v46 <= v45 >> 3 )
              {
LABEL_36:
                sub_9BAAD0(v56, v45);
                sub_2ABDB00(v56, &v62, v63);
                v44 = v62;
                v46 = *(_DWORD *)(v16 + 56) + 1;
                v39 = v63[0];
              }
              *(_DWORD *)(v16 + 56) = v46;
              if ( *v39 != -4096 )
                --*(_DWORD *)(v16 + 60);
              *v39 = v44;
              v43 = 0;
              *((_DWORD *)v39 + 2) = 0;
              goto LABEL_28;
            }
LABEL_35:
            v45 *= 2;
            goto LABEL_36;
          }
LABEL_27:
          v43 = *(_DWORD *)(v41 + 8);
LABEL_28:
          if ( v43 == v28 )
          {
            v35 = *(_QWORD *)(v35 + 8);
            if ( v35 != v36 )
              continue;
          }
          return 0;
        }
        v63[0] = 0;
        ++*(_QWORD *)(v16 + 40);
        goto LABEL_35;
      }
    }
    return 0;
  }
LABEL_22:
  v34 = *(_DWORD *)(v32 + 8);
  result = 1;
  if ( v34 <= v28 )
    goto LABEL_23;
  return result;
}
