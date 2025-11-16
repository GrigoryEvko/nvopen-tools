// Function: sub_1DCD430
// Address: 0x1dcd430
//
__int64 __fastcall sub_1DCD430(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v7; // rcx
  _WORD *v8; // r13
  unsigned __int16 v10; // r12
  unsigned int v11; // r11d
  __int16 *v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rbx
  unsigned int v15; // esi
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int16 v20; // ax
  __int64 v21; // rax
  unsigned int v22; // r14d
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned int v25; // edx
  int v26; // r13d
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r11
  unsigned __int16 *v30; // rsi
  unsigned __int16 *v31; // rsi
  __int64 v32; // rdx
  unsigned __int16 *v33; // rax
  int v34; // ecx
  unsigned __int16 v36; // r14
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int16 v39; // ax
  int v40; // eax
  int v41; // eax
  int v42; // esi
  int v43; // esi
  __int64 v44; // rdi
  int v45; // r9d
  __int64 v46; // r10
  int v47; // esi
  int v48; // esi
  int v49; // r9d
  __int64 v50; // rdi
  unsigned int v51; // [rsp+8h] [rbp-68h]
  unsigned int v52; // [rsp+8h] [rbp-68h]
  __int64 v53; // [rsp+10h] [rbp-60h]
  int v54; // [rsp+10h] [rbp-60h]
  unsigned int v55; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+20h] [rbp-50h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  unsigned int v60; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v61[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = *(_QWORD *)(a1 + 360);
  v60 = 0;
  if ( !v5 )
    BUG();
  v7 = *(unsigned int *)(*(_QWORD *)(v5 + 8) + 24LL * a2 + 4);
  v8 = (_WORD *)(*(_QWORD *)(v5 + 56) + 2 * v7);
  if ( !*v8 )
    return 0;
  v10 = a2 + *v8;
  v11 = 0;
  v59 = 0;
  v12 = v8 + 1;
  v57 = a1 + 440;
  do
  {
    v13 = *(_QWORD *)(a1 + 368);
    v14 = *(_QWORD *)(v13 + 8LL * v10);
    if ( !v14 )
      goto LABEL_9;
    v15 = *(_DWORD *)(a1 + 464);
    if ( v15 )
    {
      v16 = *(_QWORD *)(a1 + 448);
      v13 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
      LODWORD(a5) = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v17 = v16 + 16LL * (unsigned int)a5;
      v18 = *(_QWORD *)v17;
      if ( v14 == *(_QWORD *)v17 )
      {
LABEL_7:
        v19 = *(_DWORD *)(v17 + 8);
        if ( v11 < v19 )
        {
          v13 = v10;
          v59 = v14;
          v11 = v19;
          v60 = v10;
        }
        goto LABEL_9;
      }
      v54 = 1;
      v7 = 0;
      while ( v18 != -8 )
      {
        if ( !v7 && v18 == -16 )
          v7 = v17;
        LODWORD(a5) = (v15 - 1) & (v54 + a5);
        v17 = v16 + 16LL * (unsigned int)a5;
        v18 = *(_QWORD *)v17;
        if ( v14 == *(_QWORD *)v17 )
          goto LABEL_7;
        ++v54;
      }
      if ( !v7 )
        v7 = v17;
      v40 = *(_DWORD *)(a1 + 456);
      ++*(_QWORD *)(a1 + 440);
      v41 = v40 + 1;
      if ( 4 * v41 < 3 * v15 )
      {
        LODWORD(a5) = v15 >> 3;
        if ( v15 - *(_DWORD *)(a1 + 460) - v41 > v15 >> 3 )
          goto LABEL_36;
        v52 = v11;
        sub_1DC6D40(v57, v15);
        v47 = *(_DWORD *)(a1 + 464);
        if ( !v47 )
        {
LABEL_66:
          ++*(_DWORD *)(a1 + 456);
          BUG();
        }
        v48 = v47 - 1;
        a5 = *(_QWORD *)(a1 + 448);
        v46 = 0;
        v11 = v52;
        v49 = 1;
        v13 = v48 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v41 = *(_DWORD *)(a1 + 456) + 1;
        v7 = a5 + 16 * v13;
        v50 = *(_QWORD *)v7;
        if ( v14 == *(_QWORD *)v7 )
          goto LABEL_36;
        while ( v50 != -8 )
        {
          if ( v50 == -16 && !v46 )
            v46 = v7;
          v13 = v48 & (unsigned int)(v49 + v13);
          v7 = a5 + 16LL * (unsigned int)v13;
          v50 = *(_QWORD *)v7;
          if ( v14 == *(_QWORD *)v7 )
            goto LABEL_36;
          ++v49;
        }
        goto LABEL_44;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 440);
    }
    v55 = v11;
    sub_1DC6D40(v57, 2 * v15);
    v42 = *(_DWORD *)(a1 + 464);
    if ( !v42 )
      goto LABEL_66;
    v43 = v42 - 1;
    a5 = *(_QWORD *)(a1 + 448);
    v11 = v55;
    v13 = v43 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v41 = *(_DWORD *)(a1 + 456) + 1;
    v7 = a5 + 16 * v13;
    v44 = *(_QWORD *)v7;
    if ( v14 == *(_QWORD *)v7 )
      goto LABEL_36;
    v45 = 1;
    v46 = 0;
    while ( v44 != -8 )
    {
      if ( !v46 && v44 == -16 )
        v46 = v7;
      v13 = v43 & (unsigned int)(v45 + v13);
      v7 = a5 + 16LL * (unsigned int)v13;
      v44 = *(_QWORD *)v7;
      if ( v14 == *(_QWORD *)v7 )
        goto LABEL_36;
      ++v45;
    }
LABEL_44:
    if ( v46 )
      v7 = v46;
LABEL_36:
    *(_DWORD *)(a1 + 456) = v41;
    if ( *(_QWORD *)v7 != -8 )
      --*(_DWORD *)(a1 + 460);
    *(_QWORD *)v7 = v14;
    *(_DWORD *)(v7 + 8) = 0;
LABEL_9:
    v20 = *v12++;
    v10 += v20;
  }
  while ( v20 );
  if ( v59 )
  {
    sub_1D041C0(a3, &v60, v13, v7, a5);
    v21 = *(unsigned int *)(v59 + 40);
    if ( (_DWORD)v21 )
    {
      v53 = a1;
      v22 = a2;
      v23 = 0;
      v58 = 40 * v21;
      do
      {
        while ( 1 )
        {
          v24 = v23 + *(_QWORD *)(v59 + 32);
          if ( !*(_BYTE *)v24 && (*(_BYTE *)(v24 + 3) & 0x10) != 0 )
          {
            v25 = *(_DWORD *)(v24 + 8);
            if ( v25 )
            {
              v26 = *(_DWORD *)(v24 + 8);
              v27 = *(_QWORD *)(v53 + 360);
              v28 = *(_QWORD *)(v27 + 56);
              v29 = *(_QWORD *)(v27 + 8) + 24LL * v25;
              v30 = (unsigned __int16 *)(v28 + 2LL * *(unsigned int *)(v29 + 8));
              LODWORD(v27) = *v30;
              v31 = v30 + 1;
              v32 = (unsigned int)v27 + v25;
              if ( !(_WORD)v27 )
                v31 = 0;
LABEL_18:
              v33 = v31;
              if ( v31 )
                break;
            }
          }
LABEL_22:
          v23 += 40;
          if ( v23 == v58 )
            return v59;
        }
        while ( v22 != (unsigned __int16)v32 )
        {
          v34 = *v33;
          v31 = 0;
          ++v33;
          if ( !(_WORD)v34 )
            goto LABEL_18;
          v32 = (unsigned int)(v34 + v32);
          if ( !v33 )
            goto LABEL_22;
        }
        v51 = v22;
        v36 = v26;
        v37 = v28 + 2LL * *(unsigned int *)(v29 + 4);
        while ( 1 )
        {
          v38 = v37;
          if ( !v37 )
            break;
          while ( 1 )
          {
            v38 += 2;
            v61[0] = v36;
            sub_1D041C0(a3, v61, v32, v37, v28);
            v39 = *(_WORD *)(v38 - 2);
            v37 = 0;
            if ( !v39 )
              break;
            v36 += v39;
            if ( !v38 )
              goto LABEL_28;
          }
        }
LABEL_28:
        v22 = v51;
        v23 += 40;
      }
      while ( v23 != v58 );
    }
  }
  return v59;
}
