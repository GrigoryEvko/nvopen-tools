// Function: sub_AD4CC0
// Address: 0xad4cc0
//
__int64 __fastcall sub_AD4CC0(unsigned __int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdi
  int v8; // r15d
  __int64 v9; // rcx
  _QWORD *v10; // r9
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // r11
  unsigned __int64 v14; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  int v20; // edx
  unsigned __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 *v25; // rsi
  __int64 v26; // rcx
  int v27; // r10d
  __int64 *v28; // r8
  __int64 v29; // rdx
  _QWORD *v30; // r12
  __int64 v31; // rax
  unsigned __int64 *v32; // r12
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  int v36; // esi
  __int64 v37; // r8
  int v38; // esi
  unsigned int v39; // ecx
  __int64 *v40; // rdx
  __int64 v41; // r9
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rcx
  int v45; // eax
  int v46; // eax
  int v47; // eax
  int v48; // r8d
  unsigned int v49; // r14d
  _QWORD *v50; // rdi
  _BYTE *v51; // rcx
  int v52; // edx
  int v53; // r10d
  int v54; // r10d
  _QWORD *v55; // r8
  __int64 v56; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v57; // [rsp+8h] [rbp-38h] BYREF

  if ( *a3 > 3u )
    goto LABEL_13;
  v5 = sub_BD5C60(a1, a2, a3);
  v6 = *(_QWORD *)v5;
  a2 = *(unsigned int *)(*(_QWORD *)v5 + 2048LL);
  v7 = *(_QWORD *)v5 + 2024LL;
  if ( !(_DWORD)a2 )
  {
    ++*(_QWORD *)(v6 + 2024);
    goto LABEL_8;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v6 + 2032);
  v10 = 0;
  v11 = (a2 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (_QWORD *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( a3 != (_BYTE *)*v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = v12;
      v11 = (a2 - 1) & (v8 + v11);
      v12 = (_QWORD *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( a3 == (_BYTE *)*v12 )
        goto LABEL_4;
      ++v8;
    }
    if ( !v10 )
      v10 = v12;
    v45 = *(_DWORD *)(v6 + 2040);
    ++*(_QWORD *)(v6 + 2024);
    v20 = v45 + 1;
    if ( 4 * (v45 + 1) < (unsigned int)(3 * a2) )
    {
      if ( (int)a2 - *(_DWORD *)(v6 + 2044) - v20 > (unsigned int)a2 >> 3 )
      {
LABEL_10:
        *(_DWORD *)(v6 + 2040) = v20;
        if ( *v10 != -4096 )
          --*(_DWORD *)(v6 + 2044);
        *v10 = a3;
        v10[1] = 0;
LABEL_13:
        v21 = (unsigned __int64)a3;
        if ( sub_AC30F0((__int64)a3) )
          return v21;
        v22 = sub_BD3BE0(a3);
        v24 = *(_QWORD *)sub_BD5C60(a1, a2, v23);
        v56 = v22;
        v25 = (__int64 *)*(unsigned int *)(v24 + 2048);
        if ( (_DWORD)v25 )
        {
          v26 = *(_QWORD *)(v24 + 2032);
          v27 = 1;
          v28 = 0;
          v29 = ((_DWORD)v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v30 = (_QWORD *)(v26 + 16 * v29);
          v31 = *v30;
          if ( v22 == *v30 )
          {
LABEL_17:
            v32 = v30 + 1;
LABEL_18:
            v21 = *v32;
            if ( *v32 )
              return sub_AD4C90(*v32, *(__int64 ***)(a1 + 8), 0);
            v33 = (__int64 *)sub_BD5C60(a1, v25, v29);
            v34 = *(_QWORD *)(a1 - 32);
            v35 = *v33;
            v36 = *(_DWORD *)(v35 + 2048);
            v37 = *(_QWORD *)(v35 + 2032);
            if ( v36 )
            {
              v38 = v36 - 1;
              v39 = v38 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
              v40 = (__int64 *)(v37 + 16LL * v39);
              v41 = *v40;
              if ( v34 == *v40 )
              {
LABEL_22:
                *v40 = -8192;
                --*(_DWORD *)(v35 + 2040);
                ++*(_DWORD *)(v35 + 2044);
              }
              else
              {
                v52 = 1;
                while ( v41 != -4096 )
                {
                  v53 = v52 + 1;
                  v39 = v38 & (v52 + v39);
                  v40 = (__int64 *)(v37 + 16LL * v39);
                  v41 = *v40;
                  if ( v34 == *v40 )
                    goto LABEL_22;
                  v52 = v53;
                }
              }
            }
            *v32 = a1;
            sub_AC2B30(a1 - 32, v22);
            v42 = *(_QWORD *)(v22 + 8);
            if ( v42 != *(_QWORD *)(a1 + 8) )
              *(_QWORD *)(a1 + 8) = v42;
            return v21;
          }
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v28 )
              v28 = v30;
            v29 = ((_DWORD)v25 - 1) & (unsigned int)(v27 + v29);
            v30 = (_QWORD *)(v26 + 16LL * (unsigned int)v29);
            v31 = *v30;
            if ( v22 == *v30 )
              goto LABEL_17;
            ++v27;
          }
          v43 = *(_DWORD *)(v24 + 2040);
          if ( !v28 )
            v28 = v30;
          ++*(_QWORD *)(v24 + 2024);
          v29 = (unsigned int)(v43 + 1);
          v57 = v28;
          if ( 4 * (int)v29 < (unsigned int)(3 * (_DWORD)v25) )
          {
            v44 = v22;
            if ( (int)v25 - *(_DWORD *)(v24 + 2044) - (int)v29 > (unsigned int)v25 >> 3 )
            {
LABEL_35:
              *(_DWORD *)(v24 + 2040) = v29;
              if ( *v28 != -4096 )
                --*(_DWORD *)(v24 + 2044);
              *v28 = v44;
              v32 = (unsigned __int64 *)(v28 + 1);
              v28[1] = 0;
              goto LABEL_18;
            }
LABEL_56:
            sub_ACC500(v24 + 2024, (int)v25);
            v25 = &v56;
            sub_AC7020(v24 + 2024, &v56, &v57);
            v44 = v56;
            v28 = v57;
            v29 = (unsigned int)(*(_DWORD *)(v24 + 2040) + 1);
            goto LABEL_35;
          }
        }
        else
        {
          ++*(_QWORD *)(v24 + 2024);
          v57 = 0;
        }
        LODWORD(v25) = 2 * (_DWORD)v25;
        goto LABEL_56;
      }
      sub_ACC500(v7, a2);
      v46 = *(_DWORD *)(v6 + 2048);
      if ( v46 )
      {
        v47 = v46 - 1;
        a2 = *(_QWORD *)(v6 + 2032);
        v48 = 1;
        v49 = v47 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v20 = *(_DWORD *)(v6 + 2040) + 1;
        v50 = 0;
        v10 = (_QWORD *)(a2 + 16LL * v49);
        v51 = (_BYTE *)*v10;
        if ( a3 != (_BYTE *)*v10 )
        {
          while ( v51 != (_BYTE *)-4096LL )
          {
            if ( !v50 && v51 == (_BYTE *)-8192LL )
              v50 = v10;
            v49 = v47 & (v48 + v49);
            v10 = (_QWORD *)(a2 + 16LL * v49);
            v51 = (_BYTE *)*v10;
            if ( a3 == (_BYTE *)*v10 )
              goto LABEL_10;
            ++v48;
          }
          if ( v50 )
            v10 = v50;
        }
        goto LABEL_10;
      }
LABEL_76:
      ++*(_DWORD *)(v6 + 2040);
      BUG();
    }
LABEL_8:
    sub_ACC500(v7, 2 * a2);
    v16 = *(_DWORD *)(v6 + 2048);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v6 + 2032);
      v19 = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v20 = *(_DWORD *)(v6 + 2040) + 1;
      v10 = (_QWORD *)(v18 + 16LL * v19);
      a2 = *v10;
      if ( a3 != (_BYTE *)*v10 )
      {
        v54 = 1;
        v55 = 0;
        while ( a2 != -4096 )
        {
          if ( !v55 && a2 == -8192 )
            v55 = v10;
          v19 = v17 & (v54 + v19);
          v10 = (_QWORD *)(v18 + 16LL * v19);
          a2 = *v10;
          if ( a3 == (_BYTE *)*v10 )
            goto LABEL_10;
          ++v54;
        }
        if ( v55 )
          v10 = v55;
      }
      goto LABEL_10;
    }
    goto LABEL_76;
  }
LABEL_4:
  v14 = v12[1];
  if ( !v14 )
    goto LABEL_13;
  return sub_AD4C90(v14, *(__int64 ***)(a1 + 8), 0);
}
