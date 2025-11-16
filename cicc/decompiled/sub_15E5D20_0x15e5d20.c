// Function: sub_15E5D20
// Address: 0x15e5d20
//
void __fastcall sub_15E5D20(__int64 a1, const void *a2, size_t a3)
{
  __int64 v5; // r14
  unsigned int v6; // edx
  size_t **v7; // r10
  size_t *v8; // rbx
  size_t v9; // r14
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned int v13; // esi
  __int64 v14; // r9
  __int64 v15; // rdi
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned int v22; // r9d
  __int64 *v23; // r10
  __int64 v24; // r8
  _BYTE *v25; // rdi
  size_t **v26; // rax
  size_t **v27; // rax
  int v28; // r11d
  _QWORD *v29; // r10
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // rax
  _BYTE *v33; // rax
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // edx
  __int64 v38; // rdi
  int v39; // r10d
  _QWORD *v40; // r9
  int v41; // eax
  int v42; // edx
  __int64 v43; // rdi
  _QWORD *v44; // r8
  unsigned int v45; // r15d
  int v46; // r9d
  __int64 v47; // rsi
  __int64 *v48; // [rsp+8h] [rbp-58h]
  __int64 *v49; // [rsp+10h] [rbp-50h]
  __int64 *v50; // [rsp+10h] [rbp-50h]
  unsigned int v51; // [rsp+10h] [rbp-50h]
  unsigned int v52; // [rsp+18h] [rbp-48h]
  unsigned int v53; // [rsp+18h] [rbp-48h]
  __int64 v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+20h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 34) & 0x20) == 0 )
  {
    if ( !a3 )
      return;
LABEL_3:
    v5 = *(_QWORD *)sub_16498A0(a1);
    v6 = sub_16D19C0(v5 + 2800, a2, a3);
    v7 = (size_t **)(*(_QWORD *)(v5 + 2800) + 8LL * v6);
    v8 = *v7;
    if ( *v7 )
    {
      if ( v8 != (size_t *)-8LL )
      {
LABEL_5:
        v9 = *v8;
        v10 = v8 + 2;
        goto LABEL_7;
      }
      --*(_DWORD *)(v5 + 2816);
    }
    v49 = (__int64 *)v7;
    v52 = v6;
    v21 = malloc(a3 + 17);
    v22 = v52;
    v23 = v49;
    v24 = v21;
    if ( !v21 )
    {
      if ( a3 == -17 )
      {
        v32 = malloc(1u);
        v24 = 0;
        v22 = v52;
        v23 = v49;
        if ( v32 )
        {
          v25 = (_BYTE *)(v32 + 16);
          v24 = v32;
          goto LABEL_33;
        }
      }
      v48 = v23;
      v51 = v22;
      v54 = v24;
      sub_16BD1C0("Allocation failed");
      v24 = v54;
      v22 = v51;
      v23 = v48;
    }
    v25 = (_BYTE *)(v24 + 16);
    if ( a3 + 1 <= 1 )
    {
LABEL_16:
      v25[a3] = 0;
      *(_QWORD *)v24 = a3;
      *(_BYTE *)(v24 + 8) = 0;
      *v23 = v24;
      ++*(_DWORD *)(v5 + 2812);
      v26 = (size_t **)(*(_QWORD *)(v5 + 2800) + 8LL * (unsigned int)sub_16D1CD0(v5 + 2800, v22));
      v8 = *v26;
      if ( *v26 == (size_t *)-8LL || !v8 )
      {
        v27 = v26 + 1;
        do
        {
          do
            v8 = *v27++;
          while ( !v8 );
        }
        while ( v8 == (size_t *)-8LL );
      }
      goto LABEL_5;
    }
LABEL_33:
    v50 = v23;
    v53 = v22;
    v55 = v24;
    v33 = memcpy(v25, a2, a3);
    v23 = v50;
    v22 = v53;
    v24 = v55;
    v25 = v33;
    goto LABEL_16;
  }
  v10 = a2;
  v9 = a3;
  if ( a3 )
    goto LABEL_3;
LABEL_7:
  v11 = sub_16498A0(a1);
  v12 = *(_QWORD *)v11;
  v13 = *(_DWORD *)(*(_QWORD *)v11 + 2792LL);
  v14 = *(_QWORD *)v11 + 2768LL;
  if ( !v13 )
  {
    ++*(_QWORD *)(v12 + 2768);
    goto LABEL_40;
  }
  v15 = *(_QWORD *)(v12 + 2776);
  v16 = (v13 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v17 = (_QWORD *)(v15 + 24LL * v16);
  v18 = *v17;
  if ( a1 != *v17 )
  {
    v28 = 1;
    v29 = 0;
    while ( v18 != -8 )
    {
      if ( !v29 && v18 == -16 )
        v29 = v17;
      v16 = (v13 - 1) & (v28 + v16);
      v17 = (_QWORD *)(v15 + 24LL * v16);
      v18 = *v17;
      if ( a1 == *v17 )
        goto LABEL_9;
      ++v28;
    }
    v30 = *(_DWORD *)(v12 + 2784);
    if ( v29 )
      v17 = v29;
    ++*(_QWORD *)(v12 + 2768);
    v31 = v30 + 1;
    if ( 4 * v31 < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(v12 + 2788) - v31 > v13 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(v12 + 2784) = v31;
        if ( *v17 != -8 )
          --*(_DWORD *)(v12 + 2788);
        *v17 = a1;
        v17[1] = 0;
        v17[2] = 0;
        goto LABEL_9;
      }
      sub_15E5B50(v14, v13);
      v41 = *(_DWORD *)(v12 + 2792);
      if ( v41 )
      {
        v42 = v41 - 1;
        v43 = *(_QWORD *)(v12 + 2776);
        v44 = 0;
        v45 = (v41 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v46 = 1;
        v31 = *(_DWORD *)(v12 + 2784) + 1;
        v17 = (_QWORD *)(v43 + 24LL * v45);
        v47 = *v17;
        if ( a1 != *v17 )
        {
          while ( v47 != -8 )
          {
            if ( !v44 && v47 == -16 )
              v44 = v17;
            v45 = v42 & (v46 + v45);
            v17 = (_QWORD *)(v43 + 24LL * v45);
            v47 = *v17;
            if ( a1 == *v17 )
              goto LABEL_28;
            ++v46;
          }
          if ( v44 )
            v17 = v44;
        }
        goto LABEL_28;
      }
LABEL_65:
      sub_41A076();
    }
LABEL_40:
    sub_15E5B50(v14, 2 * v13);
    v34 = *(_DWORD *)(v12 + 2792);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v12 + 2776);
      v37 = (v34 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v31 = *(_DWORD *)(v12 + 2784) + 1;
      v17 = (_QWORD *)(v36 + 24LL * v37);
      v38 = *v17;
      if ( a1 != *v17 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( !v40 && v38 == -16 )
            v40 = v17;
          v37 = v35 & (v39 + v37);
          v17 = (_QWORD *)(v36 + 24LL * v37);
          v38 = *v17;
          if ( a1 == *v17 )
            goto LABEL_28;
          ++v39;
        }
        if ( v40 )
          v17 = v40;
      }
      goto LABEL_28;
    }
    goto LABEL_65;
  }
LABEL_9:
  v17[1] = v10;
  v17[2] = v9;
  v19 = *(_DWORD *)(a1 + 32);
  v20 = (v19 >> 15) & 0xFFFFFFBF;
  if ( v9 )
    v20 = (*(_DWORD *)(a1 + 32) >> 15) & 0xFFFFFFBF | 0x40;
  *(_DWORD *)(a1 + 32) = v19 & 0x7FFF | (v20 << 15);
}
