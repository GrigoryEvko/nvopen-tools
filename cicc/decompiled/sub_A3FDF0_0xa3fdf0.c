// Function: sub_A3FDF0
// Address: 0xa3fdf0
//
void __fastcall sub_A3FDF0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int64 a4,
        __int64 *a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int64 v7; // r13
  char *v8; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r8
  unsigned int v15; // ecx
  __int64 v16; // rdx
  __int64 v17; // r15
  unsigned __int64 v18; // r15
  char *v19; // rax
  char *v20; // rdi
  __int64 *v21; // r13
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r8
  _BYTE *v29; // rax
  unsigned int v30; // edx
  unsigned int v31; // ecx
  _QWORD *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rdx
  size_t v35; // r13
  char *v36; // rax
  __int64 v37; // rax
  int v38; // edx
  int v39; // r11d
  __int64 *v40; // r14
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r10
  __int64 *v47; // rdi
  __int128 v48; // [rsp-20h] [rbp-510h]
  _BYTE *v49; // [rsp+8h] [rbp-4E8h]
  __int64 v50; // [rsp+20h] [rbp-4D0h]
  unsigned __int64 v51; // [rsp+20h] [rbp-4D0h]
  __int64 v52; // [rsp+20h] [rbp-4D0h]
  unsigned int v53; // [rsp+2Ch] [rbp-4C4h] BYREF
  __int64 v54; // [rsp+30h] [rbp-4C0h] BYREF
  __int64 v55; // [rsp+38h] [rbp-4B8h] BYREF
  bool v56; // [rsp+47h] [rbp-4A9h] BYREF
  unsigned __int64 v57[5]; // [rsp+48h] [rbp-4A8h] BYREF
  _BYTE *v58; // [rsp+70h] [rbp-480h]
  unsigned int *v59; // [rsp+78h] [rbp-478h]
  bool *v60; // [rsp+80h] [rbp-470h]
  __int64 v61; // [rsp+90h] [rbp-460h]
  unsigned int *v62; // [rsp+98h] [rbp-458h]
  bool *v63; // [rsp+A0h] [rbp-450h]
  char *v64; // [rsp+B0h] [rbp-440h] BYREF
  __int64 v65; // [rsp+B8h] [rbp-438h]
  _BYTE v66[1072]; // [rsp+C0h] [rbp-430h] BYREF

  v64 = v66;
  v65 = 0x4000000000LL;
  v8 = *(char **)(a1 + 16);
  v55 = a1;
  v54 = a2;
  v53 = a3;
  if ( !v8 )
    return;
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v12 = *(unsigned int *)(a4 + 24);
      v13 = *((_QWORD *)v8 + 3);
      v14 = *(_QWORD *)(a4 + 8);
      if ( (_DWORD)v12 )
      {
        v12 = (unsigned int)(v12 - 1);
        v15 = v12 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v16 = v14 + 16LL * v15;
        v17 = *(_QWORD *)v16;
        if ( v13 != *(_QWORD *)v16 )
        {
          v38 = 1;
          while ( v17 != -4096 )
          {
            v39 = v38 + 1;
            v15 = v12 & (v38 + v15);
            v16 = v14 + 16LL * v15;
            v17 = *(_QWORD *)v16;
            if ( v13 == *(_QWORD *)v16 )
              goto LABEL_6;
            v38 = v39;
          }
          goto LABEL_3;
        }
LABEL_6:
        if ( *(_DWORD *)(v16 + 8) )
          break;
      }
LABEL_3:
      v8 = (char *)*((_QWORD *)v8 + 1);
      if ( !v8 )
        goto LABEL_10;
    }
    v18 = v11 | v7 & 0xFFFFFFFF00000000LL;
    v7 = v18;
    if ( v11 + 1 > (unsigned __int64)HIDWORD(v65) )
    {
      v12 = (unsigned __int64)v66;
      v51 = a4;
      sub_C8D5F0(&v64, v66, v11 + 1, 16);
      v11 = (unsigned int)v65;
      a4 = v51;
    }
    v19 = &v64[16 * v11];
    *(_QWORD *)v19 = v8;
    *((_QWORD *)v19 + 1) = v18;
    v11 = (unsigned int)(v65 + 1);
    LODWORD(v65) = v65 + 1;
    v8 = (char *)*((_QWORD *)v8 + 1);
  }
  while ( v8 );
LABEL_10:
  v20 = v64;
  if ( (unsigned int)v11 > 1 )
  {
    v56 = v53 <= *(_DWORD *)(a4 + 32);
    v21 = (__int64 *)&v64[16 * v11];
    v22 = 16 * v11;
    v57[2] = (unsigned __int64)&v53;
    v57[1] = a4;
    _BitScanReverse64(&v23, (16 * v11) >> 4);
    v57[3] = (unsigned __int64)&v56;
    *((_QWORD *)&v48 + 1) = &v53;
    *(_QWORD *)&v48 = a4;
    v49 = (_BYTE *)a4;
    v50 = (__int64)v64;
    sub_A3DDF0((__int64)v64, v21, 2LL * (int)(63 - (v23 ^ 0x3F)), v23 ^ 0x3F, v14, a6, v48, &v56);
    if ( v22 > 0x100 )
    {
      v59 = &v53;
      v40 = (__int64 *)(v50 + 256);
      v41 = v50 + 256;
      v58 = v49;
      v60 = &v56;
      sub_A3EC30(v50, (__int64 *)(v50 + 256), v24, v25, v26, v27, a7, v49);
      v46 = (__int64)v49;
      if ( v21 != (__int64 *)(v50 + 256) )
      {
        do
        {
          v47 = v40;
          v62 = &v53;
          v40 += 2;
          v61 = v46;
          v63 = &v56;
          v52 = v46;
          sub_A3EA00(v47, v41, v42, v43, v44, v45, v46, &v53, &v56);
          v46 = v52;
        }
        while ( v21 != v40 );
      }
    }
    else
    {
      v59 = &v53;
      v58 = v49;
      v60 = &v56;
      sub_A3EC30(v50, v21, v24, v25, v26, v27, a7, v49);
    }
    v28 = (unsigned int)v65;
    v20 = v64;
    v12 = (unsigned __int64)&v64[16 * (unsigned int)v65];
    if ( v64 != (char *)v12 )
    {
      v29 = v64 + 16;
      if ( (char *)v12 != v64 + 16 )
      {
        v30 = *((_DWORD *)v64 + 2);
        while ( 1 )
        {
          v31 = v30;
          v30 = *((_DWORD *)v29 + 2);
          if ( v30 < v31 )
            break;
          v29 += 16;
          if ( (_BYTE *)v12 == v29 )
            goto LABEL_28;
        }
        if ( (_BYTE *)v12 != v29 )
        {
          v57[0] = (unsigned int)v65;
          v32 = (_QWORD *)a5[1];
          if ( v32 == (_QWORD *)a5[2] )
          {
            v12 = a5[1];
            sub_A3FB50(a5, (char *)v12, &v55, &v54, v57);
            v28 = (unsigned int)v65;
            v20 = v64;
          }
          else
          {
            if ( v32 )
            {
              v33 = v54;
              v34 = v55;
              v32[2] = 0;
              v32[3] = 0;
              *v32 = v34;
              v32[1] = v33;
              v32[4] = 0;
              if ( v28 )
              {
                v35 = 4 * v28;
                v36 = (char *)sub_22077B0(4 * v28);
                v12 = 0;
                v8 = &v36[v35];
                v32[2] = v36;
                v32[4] = &v36[v35];
                memset(v36, 0, v35);
              }
              v32[3] = v8;
              v28 = (unsigned int)v65;
              v32 = (_QWORD *)a5[1];
              v20 = v64;
            }
            a5[1] = (__int64)(v32 + 5);
          }
          if ( v28 )
          {
            v37 = 0;
            do
            {
              v12 = *(unsigned int *)&v20[4 * v37 + 8];
              *(_DWORD *)(*(_QWORD *)(a5[1] - 24) + v37) = v12;
              v37 += 4;
              v20 = v64;
            }
            while ( v37 != 4 * v28 );
          }
        }
      }
    }
  }
LABEL_28:
  if ( v20 != v66 )
    _libc_free(v20, v12);
}
