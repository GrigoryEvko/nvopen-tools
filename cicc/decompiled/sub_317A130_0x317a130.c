// Function: sub_317A130
// Address: 0x317a130
//
__int64 __fastcall sub_317A130(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v4; // esi
  __int64 v6; // r9
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rax
  _BYTE *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // esi
  __int64 *v21; // r8
  __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 *v31; // r10
  __int64 v32; // rsi
  int v33; // edx
  int v34; // r11d
  int v35; // eax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rsi
  int v39; // r8d
  unsigned int v40; // r14d
  __int64 *v41; // rdi
  __int64 v42; // rcx
  int v43; // r8d
  int v44; // ecx
  __int64 v45; // rax
  int v46; // r9d
  __int64 *v47; // r8
  char v48; // [rsp+Eh] [rbp-62h]
  char v49; // [rsp+Fh] [rbp-61h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  __int64 v52; // [rsp+20h] [rbp-50h] BYREF
  __int64 v53; // [rsp+28h] [rbp-48h]
  __int64 v54; // [rsp+30h] [rbp-40h]
  __int64 v55; // [rsp+38h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (unsigned int)v2 > (unsigned int)qword_5034BA8 )
    return 0;
  v4 = *(_DWORD *)(a1 + 152);
  v6 = a1 + 128;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 128);
    goto LABEL_30;
  }
  v49 = 0;
  v7 = *(_QWORD *)(a1 + 136);
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 8LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
    goto LABEL_4;
  v34 = 1;
  v31 = 0;
  while ( v10 != -4096 )
  {
    if ( v31 || v10 != -8192 )
      v9 = v31;
    v8 = (v4 - 1) & (v34 + v8);
    v10 = *(_QWORD *)(v7 + 8LL * v8);
    if ( a2 == v10 )
    {
      v49 = 0;
      goto LABEL_4;
    }
    ++v34;
    v31 = v9;
    v9 = (__int64 *)(v7 + 8LL * v8);
  }
  v35 = *(_DWORD *)(a1 + 144);
  if ( !v31 )
    v31 = v9;
  ++*(_QWORD *)(a1 + 128);
  v33 = v35 + 1;
  if ( 4 * (v35 + 1) >= 3 * v4 )
  {
LABEL_30:
    sub_CF4090(v6, 2 * v4);
    v27 = *(_DWORD *)(a1 + 152);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 136);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = (__int64 *)(v29 + 8LL * v30);
      v32 = *v31;
      v33 = *(_DWORD *)(a1 + 144) + 1;
      if ( a2 != *v31 )
      {
        v46 = 1;
        v47 = 0;
        while ( v32 != -4096 )
        {
          if ( !v47 && v32 == -8192 )
            v47 = v31;
          v30 = v28 & (v46 + v30);
          v31 = (__int64 *)(v29 + 8LL * v30);
          v32 = *v31;
          if ( a2 == *v31 )
            goto LABEL_32;
          ++v46;
        }
        if ( v47 )
          v31 = v47;
      }
      goto LABEL_32;
    }
    goto LABEL_75;
  }
  if ( v4 - *(_DWORD *)(a1 + 148) - v33 <= v4 >> 3 )
  {
    sub_CF4090(v6, v4);
    v36 = *(_DWORD *)(a1 + 152);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 136);
      v39 = 1;
      v40 = v37 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = (__int64 *)(v38 + 8LL * v40);
      v33 = *(_DWORD *)(a1 + 144) + 1;
      v41 = 0;
      v42 = *v31;
      if ( a2 != *v31 )
      {
        while ( v42 != -4096 )
        {
          if ( !v41 && v42 == -8192 )
            v41 = v31;
          v40 = v37 & (v39 + v40);
          v31 = (__int64 *)(v38 + 8LL * v40);
          v42 = *v31;
          if ( a2 == *v31 )
            goto LABEL_32;
          ++v39;
        }
        if ( v41 )
          v31 = v41;
      }
      goto LABEL_32;
    }
LABEL_75:
    ++*(_DWORD *)(a1 + 144);
    BUG();
  }
LABEL_32:
  *(_DWORD *)(a1 + 144) = v33;
  if ( *v31 != -4096 )
    --*(_DWORD *)(a1 + 148);
  *v31 = a2;
  v49 = 1;
  v2 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
LABEL_4:
  if ( !(_DWORD)v2 )
    return 0;
  v48 = 0;
  v11 = 0;
  v51 = 8 * v2;
  v12 = a2;
  v13 = 0;
  v14 = a1;
  v15 = v12;
  do
  {
    v16 = *(_QWORD *)(v15 - 8);
    v17 = *(_BYTE **)(v16 + 4 * v13);
    if ( !v17 )
      BUG();
    if ( *v17 > 0x1Cu )
    {
      if ( (_BYTE *)v15 == v17 )
        goto LABEL_10;
      v50 = *(_QWORD *)(32LL * *(unsigned int *)(v15 + 72) + v16 + v13);
      if ( !(unsigned __int8)sub_2A64220(*(__int64 **)(v14 + 56), v50) )
        goto LABEL_10;
      v18 = *(unsigned int *)(v14 + 120);
      v19 = *(_QWORD *)(v14 + 104);
      if ( (_DWORD)v18 )
      {
        v20 = (v18 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
        v21 = (__int64 *)(v19 + 8LL * v20);
        v22 = *v21;
        if ( v50 == *v21 )
        {
LABEL_18:
          if ( v21 != (__int64 *)(v19 + 8 * v18) )
            goto LABEL_10;
        }
        else
        {
          v43 = 1;
          while ( v22 != -4096 )
          {
            v44 = v43 + 1;
            v20 = (v18 - 1) & (v43 + v20);
            v21 = (__int64 *)(v19 + 8LL * v20);
            v22 = *v21;
            if ( v50 == *v21 )
              goto LABEL_18;
            v43 = v44;
          }
        }
      }
    }
    v23 = sub_31751A0(v14, v17);
    if ( v23 )
    {
      if ( v11 )
      {
        if ( v23 != v11 )
          return 0;
      }
      else
      {
        v11 = v23;
      }
    }
    else
    {
      if ( v49 )
      {
        v45 = *(unsigned int *)(v14 + 168);
        if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(v14 + 172) )
        {
          sub_C8D5F0(v14 + 160, (const void *)(v14 + 176), v45 + 1, 8u, v24, v25);
          v45 = *(unsigned int *)(v14 + 168);
        }
        *(_QWORD *)(*(_QWORD *)(v14 + 160) + 8 * v45) = v15;
        ++*(_DWORD *)(v14 + 168);
        return 0;
      }
      if ( *v17 != 84 )
        return 0;
      v48 = 1;
    }
LABEL_10:
    v13 += 8;
  }
  while ( v51 != v13 );
  if ( !v11 )
    return 0;
  if ( v48 )
  {
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    if ( !(unsigned __int8)sub_3179C40(v14, v11, v15, (__int64)&v52) )
      v11 = 0;
    sub_C7D6A0(v53, 8LL * (unsigned int)v55, 8);
  }
  return v11;
}
