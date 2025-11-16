// Function: sub_153F240
// Address: 0x153f240
//
void __fastcall sub_153F240(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  char *v7; // r13
  unsigned int v10; // r14d
  int v11; // r12d
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  char **v20; // rax
  _BYTE *v21; // rdi
  bool v22; // dl
  __int64 *v23; // r12
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r9
  _BYTE *v30; // rsi
  _BYTE *v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // ecx
  _QWORD *v34; // r12
  __int64 v35; // rax
  __int64 v36; // rdx
  size_t v37; // r14
  char *v38; // rax
  __int64 v39; // rax
  int v40; // eax
  int v41; // r10d
  __int64 *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdi
  __int128 v48; // [rsp-20h] [rbp-510h]
  __int64 v49; // [rsp+8h] [rbp-4E8h]
  unsigned int v50; // [rsp+2Ch] [rbp-4C4h] BYREF
  __int64 v51; // [rsp+30h] [rbp-4C0h] BYREF
  __int64 v52; // [rsp+38h] [rbp-4B8h] BYREF
  bool v53; // [rsp+47h] [rbp-4A9h] BYREF
  unsigned __int64 v54[5]; // [rsp+48h] [rbp-4A8h] BYREF
  __int64 v55; // [rsp+70h] [rbp-480h]
  unsigned int *v56; // [rsp+78h] [rbp-478h]
  bool *v57; // [rsp+80h] [rbp-470h]
  __int64 v58; // [rsp+90h] [rbp-460h]
  unsigned int *v59; // [rsp+98h] [rbp-458h]
  bool *v60; // [rsp+A0h] [rbp-450h]
  _BYTE *v61; // [rsp+B0h] [rbp-440h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-438h]
  _BYTE v63[1072]; // [rsp+C0h] [rbp-430h] BYREF

  v52 = a1;
  v62 = 0x4000000000LL;
  v7 = *(char **)(a1 + 8);
  v51 = a2;
  v50 = a3;
  v61 = v63;
  if ( v7 )
  {
    v10 = 0;
    do
    {
      v11 = *(_DWORD *)(a4 + 24);
      if ( v11 )
      {
        v12 = v11 - 1;
        v13 = sub_1648700(v7);
        v14 = *(_QWORD *)(a4 + 8);
        v15 = v13;
        v16 = v12 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v15 == *v17 )
        {
LABEL_5:
          if ( *((_DWORD *)v17 + 2) )
          {
            v19 = v10;
            if ( v10 >= HIDWORD(v62) )
            {
              sub_16CD150(&v61, v63, 0, 16);
              v19 = (unsigned int)v62;
            }
            v20 = (char **)&v61[16 * v19];
            *v20 = v7;
            v20[1] = (char *)v10;
            v10 = v62 + 1;
            LODWORD(v62) = v62 + 1;
          }
        }
        else
        {
          v40 = 1;
          while ( v18 != -8 )
          {
            v41 = v40 + 1;
            v16 = v12 & (v40 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v15 == *v17 )
              goto LABEL_5;
            v40 = v41;
          }
        }
      }
      v7 = (char *)*((_QWORD *)v7 + 1);
    }
    while ( v7 );
    v21 = v61;
    if ( v10 > 1 )
    {
      v22 = 0;
      if ( v50 <= *(_DWORD *)(a4 + 36) )
        v22 = v50 > *(_DWORD *)(a4 + 32);
      v53 = v22;
      v23 = (__int64 *)&v61[16 * v10];
      v54[3] = (unsigned __int64)&v53;
      _BitScanReverse64(&v24, (16LL * v10) >> 4);
      v54[2] = (unsigned __int64)&v50;
      *((_QWORD *)&v48 + 1) = &v50;
      *(_QWORD *)&v48 = a4;
      v49 = (__int64)v61;
      v54[1] = a4;
      sub_153DA20((__int64)v61, v23, 2LL * (int)(63 - (v24 ^ 0x3F)), v24 ^ 0x3F, a5, a6, v48, (__int64)&v53);
      if ( 16 * (unsigned __int64)v10 > 0x100 )
      {
        v42 = (__int64 *)(v49 + 256);
        v55 = a4;
        v57 = &v53;
        v56 = &v50;
        sub_153D780(v49, (__int64 *)(v49 + 256), v25, v26, v27, v28, a7);
        if ( v23 != (__int64 *)(v49 + 256) )
        {
          do
          {
            v47 = (__int64)v42;
            v42 += 2;
            v58 = a4;
            v59 = &v50;
            v60 = &v53;
            sub_153D720(v47, v49 + 256, v43, v44, v45, v46, a4);
          }
          while ( v23 != v42 );
        }
      }
      else
      {
        v55 = a4;
        v56 = &v50;
        v57 = &v53;
        sub_153D780(v49, v23, v25, v26, v27, v28, a7);
      }
      v29 = (unsigned int)v62;
      v21 = v61;
      v30 = &v61[16 * (unsigned int)v62];
      if ( v30 != v61 )
      {
        v31 = v61 + 16;
        if ( v30 != v61 + 16 )
        {
          v32 = *((_DWORD *)v61 + 2);
          while ( 1 )
          {
            v33 = v32;
            v32 = *((_DWORD *)v31 + 2);
            if ( v32 < v33 )
              break;
            v31 += 16;
            if ( v30 == v31 )
              goto LABEL_31;
          }
          if ( v30 != v31 )
          {
            v54[0] = (unsigned int)v62;
            v34 = *(_QWORD **)(a5 + 8);
            if ( v34 == *(_QWORD **)(a5 + 16) )
            {
              sub_153EFA0((__int64 *)a5, *(char **)(a5 + 8), &v52, &v51, v54);
              v29 = (unsigned int)v62;
              v21 = v61;
            }
            else
            {
              if ( v34 )
              {
                v35 = v51;
                v36 = v52;
                v34[2] = 0;
                v37 = 4 * v29;
                v34[3] = 0;
                *v34 = v36;
                v34[1] = v35;
                v34[4] = 0;
                if ( v29 )
                {
                  v38 = (char *)sub_22077B0(4 * v29);
                  v7 = &v38[v37];
                  v34[2] = v38;
                  v34[4] = &v38[v37];
                  if ( &v38[v37] != v38 )
                    memset(v38, 0, v37);
                }
                v34[3] = v7;
                v29 = (unsigned int)v62;
                v34 = *(_QWORD **)(a5 + 8);
                v21 = v61;
              }
              *(_QWORD *)(a5 + 8) = v34 + 5;
            }
            if ( v29 )
            {
              v39 = 0;
              do
              {
                *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 8) - 24LL) + v39) = *(_DWORD *)&v21[4 * v39 + 8];
                v39 += 4;
                v21 = v61;
              }
              while ( v39 != 4 * v29 );
            }
          }
        }
      }
    }
LABEL_31:
    if ( v21 != v63 )
      _libc_free((unsigned __int64)v21);
  }
}
