// Function: sub_2A12AD0
// Address: 0x2a12ad0
//
__int64 *__fastcall sub_2A12AD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v7; // eax
  __int64 v8; // rcx
  int v9; // esi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 *v13; // r13
  char v14; // di
  __int64 v15; // r8
  int v16; // esi
  unsigned int v17; // edx
  __int64 **v18; // rax
  __int64 *v19; // r9
  __int64 *v20; // rdi
  __int64 **v21; // r15
  unsigned int v23; // esi
  unsigned int v24; // eax
  __int64 **v25; // rcx
  int v26; // edx
  unsigned int v27; // r8d
  __int64 v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rdi
  int v32; // r8d
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // r9
  __int64 v36; // rdi
  _BYTE *v37; // rsi
  int v38; // r8d
  _BYTE *v39; // rsi
  int v40; // eax
  int v41; // r11d
  int v42; // r9d
  __int64 v43; // rsi
  int v44; // edx
  unsigned int v45; // eax
  __int64 *v46; // rdi
  int v47; // eax
  int v48; // r10d
  __int64 v49; // rsi
  int v50; // edx
  unsigned int v51; // eax
  __int64 *v52; // rdi
  int v53; // r9d
  __int64 **v54; // r8
  int v55; // edx
  int v56; // edx
  int v57; // r9d
  _QWORD v58[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a3 + 24);
  v8 = *(_QWORD *)(a3 + 8);
  if ( v7 )
  {
    v9 = v7 - 1;
    v10 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( a1 == *v11 )
    {
LABEL_3:
      v13 = (__int64 *)v11[1];
      goto LABEL_4;
    }
    v40 = 1;
    while ( v12 != -4096 )
    {
      v42 = v40 + 1;
      v10 = v9 & (v40 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( a1 == *v11 )
        goto LABEL_3;
      v40 = v42;
    }
  }
  v13 = 0;
LABEL_4:
  v14 = *(_BYTE *)(a4 + 8) & 1;
  if ( v14 )
  {
    v15 = a4 + 16;
    v16 = 3;
  }
  else
  {
    v23 = *(_DWORD *)(a4 + 24);
    v15 = *(_QWORD *)(a4 + 16);
    if ( !v23 )
    {
      v24 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v25 = 0;
      v26 = (v24 >> 1) + 1;
      goto LABEL_13;
    }
    v16 = v23 - 1;
  }
  v17 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v18 = (__int64 **)(v15 + 16LL * v17);
  v19 = *v18;
  if ( v13 != *v18 )
  {
    v41 = 1;
    v25 = 0;
    while ( v19 != (__int64 *)-4096LL )
    {
      if ( !v25 && v19 == (__int64 *)-8192LL )
        v25 = v18;
      v17 = v16 & (v41 + v17);
      v18 = (__int64 **)(v15 + 16LL * v17);
      v19 = *v18;
      if ( v13 == *v18 )
        goto LABEL_7;
      ++v41;
    }
    v27 = 12;
    v23 = 4;
    if ( !v25 )
      v25 = v18;
    v24 = *(_DWORD *)(a4 + 8);
    ++*(_QWORD *)a4;
    v26 = (v24 >> 1) + 1;
    if ( v14 )
    {
LABEL_14:
      if ( v27 > 4 * v26 )
      {
        if ( v23 - *(_DWORD *)(a4 + 12) - v26 > v23 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a4 + 8) = (2 * (v24 >> 1) + 2) | v24 & 1;
          if ( *v25 != (__int64 *)-4096LL )
            --*(_DWORD *)(a4 + 12);
          *v25 = v13;
          v21 = v25 + 1;
          v25[1] = 0;
LABEL_19:
          v28 = *(_QWORD *)(a3 + 56);
          *(_QWORD *)(a3 + 136) += 160LL;
          v29 = (_QWORD *)((v28 + 7) & 0xFFFFFFFFFFFFFFF8LL);
          if ( *(_QWORD *)(a3 + 64) >= (unsigned __int64)(v29 + 20) && v28 )
            *(_QWORD *)(a3 + 56) = v29 + 20;
          else
            v29 = (_QWORD *)sub_9D1E70(a3 + 56, 160, 160, 3);
          memset(v29, 0, 0xA0u);
          v29[9] = 8;
          v29[8] = v29 + 11;
          *((_BYTE *)v29 + 84) = 1;
          *v21 = v29;
          v30 = *v13;
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v31 = a4 + 16;
            v32 = 3;
          }
          else
          {
            v38 = *(_DWORD *)(a4 + 24);
            v31 = *(_QWORD *)(a4 + 16);
            if ( !v38 )
              goto LABEL_32;
            v32 = v38 - 1;
          }
          v33 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v34 = (__int64 *)(v31 + 16LL * v33);
          v35 = *v34;
          if ( v30 == *v34 )
          {
LABEL_25:
            v36 = v34[1];
            if ( v36 )
            {
              v58[0] = v29;
              *v29 = v36;
              v37 = *(_BYTE **)(v36 + 16);
              if ( v37 == *(_BYTE **)(v36 + 24) )
              {
                sub_D4C7F0(v36 + 8, v37, v58);
              }
              else
              {
                if ( v37 )
                {
                  *(_QWORD *)v37 = v58[0];
                  v37 = *(_BYTE **)(v36 + 16);
                }
                *(_QWORD *)(v36 + 16) = v37 + 8;
              }
LABEL_36:
              sub_D4F330(*v21, a2, a3);
              return v13;
            }
          }
          else
          {
            v47 = 1;
            while ( v35 != -4096 )
            {
              v48 = v47 + 1;
              v33 = v32 & (v47 + v33);
              v34 = (__int64 *)(v31 + 16LL * v33);
              v35 = *v34;
              if ( v30 == *v34 )
                goto LABEL_25;
              v47 = v48;
            }
          }
LABEL_32:
          v58[0] = v29;
          v39 = *(_BYTE **)(a3 + 40);
          if ( v39 == *(_BYTE **)(a3 + 48) )
          {
            sub_D4C7F0(a3 + 32, v39, v58);
          }
          else
          {
            if ( v39 )
            {
              *(_QWORD *)v39 = v29;
              v39 = *(_BYTE **)(a3 + 40);
            }
            *(_QWORD *)(a3 + 40) = v39 + 8;
          }
          goto LABEL_36;
        }
        sub_2A123C0(a4, v23);
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v49 = a4 + 16;
          v50 = 3;
          goto LABEL_61;
        }
        v56 = *(_DWORD *)(a4 + 24);
        v49 = *(_QWORD *)(a4 + 16);
        if ( v56 )
        {
          v50 = v56 - 1;
LABEL_61:
          v51 = v50 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v25 = (__int64 **)(v49 + 16LL * v51);
          v52 = *v25;
          if ( *v25 != v13 )
          {
            v53 = 1;
            v54 = 0;
            while ( v52 != (__int64 *)-4096LL )
            {
              if ( !v54 && v52 == (__int64 *)-8192LL )
                v54 = v25;
              v51 = v50 & (v53 + v51);
              v25 = (__int64 **)(v49 + 16LL * v51);
              v52 = *v25;
              if ( v13 == *v25 )
                goto LABEL_53;
              ++v53;
            }
LABEL_64:
            if ( v54 )
              v25 = v54;
            goto LABEL_53;
          }
          goto LABEL_53;
        }
LABEL_87:
        *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
        BUG();
      }
      sub_2A123C0(a4, 2 * v23);
      if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
      {
        v43 = a4 + 16;
        v44 = 3;
      }
      else
      {
        v55 = *(_DWORD *)(a4 + 24);
        v43 = *(_QWORD *)(a4 + 16);
        if ( !v55 )
          goto LABEL_87;
        v44 = v55 - 1;
      }
      v45 = v44 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v25 = (__int64 **)(v43 + 16LL * v45);
      v46 = *v25;
      if ( v13 != *v25 )
      {
        v57 = 1;
        v54 = 0;
        while ( v46 != (__int64 *)-4096LL )
        {
          if ( v46 == (__int64 *)-8192LL && !v54 )
            v54 = v25;
          v45 = v44 & (v57 + v45);
          v25 = (__int64 **)(v43 + 16LL * v45);
          v46 = *v25;
          if ( v13 == *v25 )
            goto LABEL_53;
          ++v57;
        }
        goto LABEL_64;
      }
LABEL_53:
      v24 = *(_DWORD *)(a4 + 8);
      goto LABEL_16;
    }
    v23 = *(_DWORD *)(a4 + 24);
LABEL_13:
    v27 = 3 * v23;
    goto LABEL_14;
  }
LABEL_7:
  v20 = v18[1];
  v21 = v18 + 1;
  if ( !v20 )
    goto LABEL_19;
  v13 = 0;
  sub_D4F330(v20, a2, a3);
  return v13;
}
