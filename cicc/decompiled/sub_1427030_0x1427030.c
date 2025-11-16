// Function: sub_1427030
// Address: 0x1427030
//
__int64 __fastcall sub_1427030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rdi
  char v7; // al
  __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // eax
  bool v14; // zf
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // esi
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // edx
  int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // rdi
  int v26; // r10d
  __int64 *v27; // r9
  char v28; // dl
  unsigned __int16 v29; // dx
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // r15
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // r8
  unsigned int v38; // ecx
  __int64 v39; // rdx
  __int64 v41; // rax
  int v42; // r11d
  __int64 *v43; // r10
  int v44; // ecx
  int v45; // eax
  int v46; // edx
  __int64 v47; // rdi
  __int64 *v48; // r8
  unsigned int v49; // r14d
  int v50; // r9d
  __int64 v51; // rsi
  __int64 v52; // [rsp+8h] [rbp-68h]
  __m128i v53[2]; // [rsp+10h] [rbp-60h] BYREF
  char v54; // [rsp+38h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v41 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v41 + 16) && (*(_BYTE *)(v41 + 33) & 0x20) != 0 && *(_DWORD *)(v41 + 36) == 4 )
      return 0;
  }
  v6 = *(_QWORD **)a1;
  v54 = 0;
  v7 = sub_13575E0(v6, a2, v53, a4);
  if ( (v7 & 2) == 0 )
  {
    if ( (v28 = *(_BYTE *)(a2 + 16), v28 != 55) && v28 != 54
      || (v29 = *(_WORD *)(a2 + 18), ((v29 >> 7) & 6) == 0) && (v29 & 1) == 0 )
    {
      if ( (v7 & 1) == 0 )
        return 0;
      v30 = sub_16498A0(a2);
      v31 = *(_QWORD *)(a2 + 40);
      v32 = v30;
      v11 = sub_1648A60(88, 1);
      if ( v11 )
      {
        v33 = sub_1643270(v32);
        sub_1648CB0(v11, v33, 21);
        v34 = *(_DWORD *)(v11 + 20);
        *(_QWORD *)(v11 + 64) = v31;
        *(_QWORD *)(v11 + 32) = 0;
        *(_QWORD *)(v11 + 72) = a2;
        *(_QWORD *)(v11 + 40) = 0;
        v14 = *(_QWORD *)(v11 - 24) == 0;
        *(_DWORD *)(v11 + 20) = v34 & 0xF0000000 | 1;
        *(_QWORD *)(v11 + 48) = 0;
        *(_QWORD *)(v11 + 24) = sub_1420030;
        *(_QWORD *)(v11 + 56) = 0;
        *(_WORD *)(v11 + 80) = 257;
        if ( !v14 )
        {
          v35 = *(_QWORD *)(v11 - 16);
          v36 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v36 = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
        }
        *(_QWORD *)(v11 - 24) = 0;
        *(_DWORD *)(v11 + 84) = -1;
LABEL_25:
        v18 = *(_DWORD *)(a1 + 48);
        v17 = a1 + 24;
        if ( v18 )
          goto LABEL_26;
LABEL_8:
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_9;
      }
LABEL_55:
      v11 = 0;
      goto LABEL_25;
    }
  }
  v8 = sub_16498A0(a2);
  v9 = *(_DWORD *)(a1 + 336);
  v10 = v8;
  v52 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a1 + 336) = v9 + 1;
  v11 = sub_1648A60(120, 1);
  if ( !v11 )
    goto LABEL_55;
  v12 = sub_1643270(v10);
  sub_1648CB0(v11, v12, 22);
  v13 = *(_DWORD *)(v11 + 20);
  *(_QWORD *)(v11 + 32) = 0;
  *(_QWORD *)(v11 + 40) = 0;
  *(_QWORD *)(v11 + 72) = a2;
  v14 = *(_QWORD *)(v11 - 24) == 0;
  *(_QWORD *)(v11 + 48) = 0;
  *(_DWORD *)(v11 + 20) = v13 & 0xF0000000 | 1;
  *(_QWORD *)(v11 + 56) = 0;
  *(_QWORD *)(v11 + 24) = sub_141FFD0;
  *(_WORD *)(v11 + 80) = 257;
  *(_QWORD *)(v11 + 64) = v52;
  if ( !v14 )
  {
    v15 = *(_QWORD *)(v11 - 16);
    v16 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v16 = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
  }
  *(_QWORD *)(v11 - 24) = 0;
  v17 = a1 + 24;
  *(_DWORD *)(v11 + 84) = v9;
  *(_DWORD *)(v11 + 88) = -1;
  *(_QWORD *)(v11 + 96) = 4;
  *(_QWORD *)(v11 + 104) = 0;
  *(_QWORD *)(v11 + 112) = 0;
  v18 = *(_DWORD *)(a1 + 48);
  if ( !v18 )
    goto LABEL_8;
LABEL_26:
  v37 = *(_QWORD *)(a1 + 32);
  v38 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = (__int64 *)(v37 + 16LL * v38);
  v39 = *v24;
  if ( a2 != *v24 )
  {
    v42 = 1;
    v43 = 0;
    while ( v39 != -8 )
    {
      if ( !v43 && v39 == -16 )
        v43 = v24;
      v38 = (v18 - 1) & (v42 + v38);
      v24 = (__int64 *)(v37 + 16LL * v38);
      v39 = *v24;
      if ( a2 == *v24 )
        goto LABEL_27;
      ++v42;
    }
    v44 = *(_DWORD *)(a1 + 40);
    if ( v43 )
      v24 = v43;
    ++*(_QWORD *)(a1 + 24);
    v23 = v44 + 1;
    if ( 4 * v23 < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(a1 + 44) - v23 > v18 >> 3 )
      {
LABEL_41:
        *(_DWORD *)(a1 + 40) = v23;
        if ( *v24 != -8 )
          --*(_DWORD *)(a1 + 44);
        *v24 = a2;
        v24[1] = 0;
        goto LABEL_27;
      }
      sub_14267C0(v17, v18);
      v45 = *(_DWORD *)(a1 + 48);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 32);
        v48 = 0;
        v49 = (v45 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v50 = 1;
        v23 = *(_DWORD *)(a1 + 40) + 1;
        v24 = (__int64 *)(v47 + 16LL * v49);
        v51 = *v24;
        if ( a2 != *v24 )
        {
          while ( v51 != -8 )
          {
            if ( !v48 && v51 == -16 )
              v48 = v24;
            v49 = v46 & (v50 + v49);
            v24 = (__int64 *)(v47 + 16LL * v49);
            v51 = *v24;
            if ( a2 == *v24 )
              goto LABEL_41;
            ++v50;
          }
          if ( v48 )
            v24 = v48;
        }
        goto LABEL_41;
      }
LABEL_66:
      ++*(_DWORD *)(a1 + 40);
      BUG();
    }
LABEL_9:
    sub_14267C0(v17, 2 * v18);
    v19 = *(_DWORD *)(a1 + 48);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 32);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 40) + 1;
      v24 = (__int64 *)(v21 + 16LL * v22);
      v25 = *v24;
      if ( a2 != *v24 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -8 )
        {
          if ( !v27 && v25 == -16 )
            v27 = v24;
          v22 = v20 & (v26 + v22);
          v24 = (__int64 *)(v21 + 16LL * v22);
          v25 = *v24;
          if ( a2 == *v24 )
            goto LABEL_41;
          ++v26;
        }
        if ( v27 )
          v24 = v27;
      }
      goto LABEL_41;
    }
    goto LABEL_66;
  }
LABEL_27:
  v24[1] = v11;
  return v11;
}
