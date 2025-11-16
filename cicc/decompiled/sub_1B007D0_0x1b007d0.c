// Function: sub_1B007D0
// Address: 0x1b007d0
//
_QWORD *__fastcall sub_1B007D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  int v8; // eax
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  char v14; // cl
  __int64 v15; // rdi
  int v16; // esi
  unsigned int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdi
  unsigned int v22; // esi
  unsigned int v23; // eax
  _QWORD *v24; // r15
  int v25; // edx
  unsigned int v26; // edi
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rdi
  int v30; // esi
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // r9
  __int64 v34; // rdi
  _BYTE *v35; // rsi
  int v36; // esi
  _BYTE *v37; // rsi
  int v38; // r10d
  int v39; // eax
  __int64 v40; // rcx
  int v41; // edx
  unsigned int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // rcx
  int v45; // edx
  unsigned int v46; // eax
  _QWORD *v47; // rsi
  int v48; // r8d
  _QWORD *v49; // rdi
  int v50; // eax
  int v51; // r10d
  int v52; // edx
  int v53; // edx
  int v54; // r9d
  int v55; // r8d
  _QWORD v56[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = 0;
  v8 = *(_DWORD *)(a3 + 24);
  if ( v8 )
  {
    v9 = *(_QWORD *)(a3 + 8);
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( a1 == *v12 )
    {
LABEL_3:
      v5 = (_QWORD *)v12[1];
    }
    else
    {
      v39 = 1;
      while ( v13 != -8 )
      {
        v54 = v39 + 1;
        v11 = v10 & (v39 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( a1 == *v12 )
          goto LABEL_3;
        v39 = v54;
      }
      v5 = 0;
    }
  }
  v14 = *(_BYTE *)(a4 + 8) & 1;
  if ( v14 )
  {
    v15 = a4 + 16;
    v16 = 3;
  }
  else
  {
    v22 = *(_DWORD *)(a4 + 24);
    v15 = *(_QWORD *)(a4 + 16);
    if ( !v22 )
    {
      v23 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v24 = 0;
      v25 = (v23 >> 1) + 1;
      goto LABEL_13;
    }
    v16 = v22 - 1;
  }
  v17 = v16 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v18 = (_QWORD *)(v15 + 16LL * v17);
  v19 = *v18;
  if ( v5 != (_QWORD *)*v18 )
  {
    v38 = 1;
    v24 = 0;
    while ( v19 != -8 )
    {
      if ( v19 == -16 && !v24 )
        v24 = v18;
      v17 = v16 & (v38 + v17);
      v18 = (_QWORD *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( v5 == (_QWORD *)*v18 )
        goto LABEL_7;
      ++v38;
    }
    v26 = 12;
    v22 = 4;
    if ( !v24 )
      v24 = v18;
    v23 = *(_DWORD *)(a4 + 8);
    ++*(_QWORD *)a4;
    v25 = (v23 >> 1) + 1;
    if ( v14 )
    {
LABEL_14:
      if ( 4 * v25 < v26 )
      {
        if ( v22 - *(_DWORD *)(a4 + 12) - v25 > v22 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a4 + 8) = (2 * (v23 >> 1) + 2) | v23 & 1;
          if ( *v24 != -8 )
            --*(_DWORD *)(a4 + 12);
          *v24 = v5;
          v24[1] = 0;
LABEL_19:
          v27 = sub_194ACF0(a3);
          v24[1] = v27;
          v28 = v27;
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v29 = a4 + 16;
            v30 = 3;
          }
          else
          {
            v36 = *(_DWORD *)(a4 + 24);
            v29 = *(_QWORD *)(a4 + 16);
            if ( !v36 )
              goto LABEL_30;
            v30 = v36 - 1;
          }
          v31 = v30 & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
          v32 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v32;
          if ( *v5 == *v32 )
          {
LABEL_22:
            v34 = v32[1];
            if ( v34 )
            {
              v56[0] = v28;
              *v28 = v34;
              v35 = *(_BYTE **)(v34 + 16);
              if ( v35 == *(_BYTE **)(v34 + 24) )
              {
                sub_13FD960(v34 + 8, v35, v56);
              }
              else
              {
                if ( v35 )
                {
                  *(_QWORD *)v35 = v56[0];
                  v35 = *(_BYTE **)(v34 + 16);
                }
                *(_QWORD *)(v34 + 16) = v35 + 8;
              }
LABEL_27:
              sub_1400330(v24[1], a2, a3);
              return v5;
            }
          }
          else
          {
            v50 = 1;
            while ( v33 != -8 )
            {
              v51 = v50 + 1;
              v31 = v30 & (v50 + v31);
              v32 = (__int64 *)(v29 + 16LL * v31);
              v33 = *v32;
              if ( *v5 == *v32 )
                goto LABEL_22;
              v50 = v51;
            }
          }
LABEL_30:
          v56[0] = v28;
          v37 = *(_BYTE **)(a3 + 40);
          if ( v37 == *(_BYTE **)(a3 + 48) )
          {
            sub_13FD960(a3 + 32, v37, v56);
          }
          else
          {
            if ( v37 )
            {
              *(_QWORD *)v37 = v28;
              v37 = *(_BYTE **)(a3 + 40);
            }
            *(_QWORD *)(a3 + 40) = v37 + 8;
          }
          goto LABEL_27;
        }
        sub_1B00120(a4, v22);
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v44 = a4 + 16;
          v45 = 3;
          goto LABEL_51;
        }
        v53 = *(_DWORD *)(a4 + 24);
        v44 = *(_QWORD *)(a4 + 16);
        if ( v53 )
        {
          v45 = v53 - 1;
LABEL_51:
          v46 = v45 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v24 = (_QWORD *)(v44 + 16LL * v46);
          v47 = (_QWORD *)*v24;
          if ( v5 != (_QWORD *)*v24 )
          {
            v48 = 1;
            v49 = 0;
            while ( v47 != (_QWORD *)-8LL )
            {
              if ( !v49 && v47 == (_QWORD *)-16LL )
                v49 = v24;
              v46 = v45 & (v48 + v46);
              v24 = (_QWORD *)(v44 + 16LL * v46);
              v47 = (_QWORD *)*v24;
              if ( v5 == (_QWORD *)*v24 )
                goto LABEL_48;
              ++v48;
            }
LABEL_54:
            if ( v49 )
              v24 = v49;
            goto LABEL_48;
          }
          goto LABEL_48;
        }
LABEL_84:
        *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
        BUG();
      }
      sub_1B00120(a4, 2 * v22);
      if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
      {
        v40 = a4 + 16;
        v41 = 3;
      }
      else
      {
        v52 = *(_DWORD *)(a4 + 24);
        v40 = *(_QWORD *)(a4 + 16);
        if ( !v52 )
          goto LABEL_84;
        v41 = v52 - 1;
      }
      v42 = v41 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v24 = (_QWORD *)(v40 + 16LL * v42);
      v43 = *v24;
      if ( v5 != (_QWORD *)*v24 )
      {
        v55 = 1;
        v49 = 0;
        while ( v43 != -8 )
        {
          if ( v43 == -16 && !v49 )
            v49 = v24;
          v42 = v41 & (v55 + v42);
          v24 = (_QWORD *)(v40 + 16LL * v42);
          v43 = *v24;
          if ( v5 == (_QWORD *)*v24 )
            goto LABEL_48;
          ++v55;
        }
        goto LABEL_54;
      }
LABEL_48:
      v23 = *(_DWORD *)(a4 + 8);
      goto LABEL_16;
    }
    v22 = *(_DWORD *)(a4 + 24);
LABEL_13:
    v26 = 3 * v22;
    goto LABEL_14;
  }
LABEL_7:
  v20 = v18[1];
  if ( !v20 )
  {
    v24 = v18;
    goto LABEL_19;
  }
  v5 = 0;
  sub_1400330(v20, a2, a3);
  return v5;
}
