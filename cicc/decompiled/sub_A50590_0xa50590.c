// Function: sub_A50590
// Address: 0xa50590
//
__int64 __fastcall sub_A50590(__int64 a1, unsigned __int64 a2, unsigned int a3, __int64 a4, __m128i a5)
{
  unsigned __int64 v5; // r13
  _QWORD *v7; // rbx
  _BYTE *v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 *v14; // rdx
  __int64 v15; // r10
  unsigned __int64 v16; // rax
  _QWORD *v17; // rcx
  _BYTE *v18; // rdi
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r10
  unsigned int v22; // esi
  __int64 *v23; // rdx
  __int64 v24; // r13
  __int64 *v25; // r13
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  _BYTE *v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // ecx
  char *v34; // r14
  size_t v35; // r13
  _DWORD *v36; // rax
  int *v37; // rdx
  int v38; // ecx
  int v40; // edx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r9
  __int64 v44; // r8
  __int64 *v45; // rbx
  __int64 v46; // rdi
  int v47; // edx
  int v48; // r14d
  __int128 v49; // [rsp-20h] [rbp-500h]
  __int64 v50; // [rsp+10h] [rbp-4D0h]
  __int64 v51; // [rsp+10h] [rbp-4D0h]
  __int64 v52; // [rsp+18h] [rbp-4C8h]
  int v53; // [rsp+18h] [rbp-4C8h]
  unsigned __int64 v54; // [rsp+18h] [rbp-4C8h]
  unsigned int v55[4]; // [rsp+2Ch] [rbp-4B4h] BYREF
  bool v56; // [rsp+3Fh] [rbp-4A1h] BYREF
  __int64 v57; // [rsp+40h] [rbp-4A0h]
  bool *v58; // [rsp+48h] [rbp-498h]
  unsigned int *v59; // [rsp+50h] [rbp-490h]
  __int64 v60; // [rsp+60h] [rbp-480h]
  bool *v61; // [rsp+68h] [rbp-478h]
  unsigned int *v62; // [rsp+70h] [rbp-470h]
  __int64 v63; // [rsp+80h] [rbp-460h]
  bool *v64; // [rsp+88h] [rbp-458h]
  unsigned int *v65; // [rsp+90h] [rbp-450h]
  _BYTE *v66; // [rsp+A0h] [rbp-440h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-438h]
  _BYTE v68[1072]; // [rsp+B0h] [rbp-430h] BYREF

  v7 = *(_QWORD **)(a2 + 16);
  v66 = v68;
  v55[0] = a3;
  v67 = 0x4000000000LL;
  if ( !v7 )
  {
    v18 = v68;
LABEL_32:
    *(_QWORD *)(a1 + 16) = 0;
    *(_OWORD *)a1 = 0;
    goto LABEL_33;
  }
  v9 = (_BYTE *)a2;
  v10 = 0;
  do
  {
    v11 = *(unsigned int *)(a4 + 24);
    v12 = v7[3];
    v13 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v11 )
    {
      a2 = ((_DWORD)v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v14 = (__int64 *)(v13 + 16 * a2);
      v15 = *v14;
      if ( v12 == *v14 )
      {
LABEL_5:
        if ( v14 != (__int64 *)(v13 + 16 * v11)
          && *(_DWORD *)(*(_QWORD *)(a4 + 32) + 16LL * *((unsigned int *)v14 + 2) + 8) )
        {
          a2 = HIDWORD(v67);
          v16 = v10 | v5 & 0xFFFFFFFF00000000LL;
          v5 = v16;
          if ( v10 + 1 > (unsigned __int64)HIDWORD(v67) )
          {
            a2 = (unsigned __int64)v68;
            v51 = a4;
            v54 = v16;
            sub_C8D5F0(&v66, v68, v10 + 1, 16);
            v10 = (unsigned int)v67;
            a4 = v51;
            v16 = v54;
          }
          v17 = &v66[16 * v10];
          *v17 = v7;
          v17[1] = v16;
          v10 = (unsigned int)(v67 + 1);
          LODWORD(v67) = v67 + 1;
        }
      }
      else
      {
        v40 = 1;
        while ( v15 != -4096 )
        {
          a2 = ((_DWORD)v11 - 1) & (unsigned int)(v40 + a2);
          v53 = v40 + 1;
          v14 = (__int64 *)(v13 + 16LL * (unsigned int)a2);
          v15 = *v14;
          if ( v12 == *v14 )
            goto LABEL_5;
          v40 = v53;
        }
      }
    }
    v7 = (_QWORD *)v7[1];
  }
  while ( v7 );
  v18 = v66;
  if ( (unsigned int)v10 <= 1 )
    goto LABEL_32;
  v19 = *v9;
  v56 = *v9 != 23;
  if ( v19 == 4 )
  {
    v20 = *(unsigned int *)(a4 + 24);
    v12 = *((_QWORD *)v9 - 4);
    v21 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v20 )
    {
      v22 = (v20 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v12 == *v23 )
      {
LABEL_15:
        if ( v23 != (__int64 *)(v21 + 16 * v20) )
        {
          LODWORD(v20) = *(_DWORD *)(*(_QWORD *)(a4 + 32) + 16LL * *((unsigned int *)v23 + 2) + 8);
          goto LABEL_17;
        }
      }
      else
      {
        v47 = 1;
        while ( v24 != -4096 )
        {
          v48 = v47 + 1;
          v22 = (v20 - 1) & (v47 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v12 == *v23 )
            goto LABEL_15;
          v47 = v48;
        }
      }
      LODWORD(v20) = 0;
    }
LABEL_17:
    v55[0] = v20;
  }
  v25 = (__int64 *)&v66[16 * v10];
  v26 = 16 * v10;
  v58 = &v56;
  v57 = a4;
  _BitScanReverse64(&v27, (16 * v10) >> 4);
  v59 = v55;
  *((_QWORD *)&v49 + 1) = &v56;
  *(_QWORD *)&v49 = a4;
  v50 = a4;
  v52 = (__int64)v66;
  sub_A4FFF0((__int64)v66, v25, 2LL * (int)(63 - (v27 ^ 0x3F)), (__int64)v55, a4, v12, v49, v55);
  if ( v26 > 0x100 )
  {
    v61 = &v56;
    v60 = v50;
    v62 = v55;
    sub_A4FD50(v52, (__int64 *)(v52 + 256), v28, (__int64)v55, v50, v30, a5);
    if ( v25 != (__int64 *)(v52 + 256) )
    {
      v44 = v50;
      v45 = (__int64 *)(v52 + 256);
      do
      {
        v64 = &v56;
        v46 = (__int64)v45;
        v45 += 2;
        v65 = v55;
        v63 = v50;
        sub_A4FCF0(v46, v52 + 256, v41, v42, v44, v43, v50);
      }
      while ( v25 != v45 );
      v7 = 0;
    }
  }
  else
  {
    v61 = &v56;
    v60 = v50;
    v62 = v55;
    sub_A4FD50(v52, v25, v28, v29, v50, v30, a5);
  }
  v18 = v66;
  a2 = (unsigned __int64)&v66[16 * (unsigned int)v67];
  if ( v66 == (_BYTE *)a2 )
    goto LABEL_32;
  v31 = v66 + 16;
  if ( (_BYTE *)a2 == v66 + 16 )
    goto LABEL_32;
  v32 = *((_DWORD *)v66 + 2);
  while ( 1 )
  {
    v33 = v32;
    v32 = *((_DWORD *)v31 + 2);
    if ( v32 < v33 )
      break;
    v31 += 16;
    if ( (_BYTE *)a2 == v31 )
      goto LABEL_32;
  }
  if ( v31 == (_BYTE *)a2 )
    goto LABEL_32;
  v34 = 0;
  if ( (_DWORD)v67 )
  {
    v35 = 4LL * (unsigned int)v67;
    a2 = 0;
    v7 = (_QWORD *)sub_22077B0(v35);
    v34 = (char *)v7 + v35;
    v36 = memset(v7, 0, v35);
    v18 = v66;
    if ( (_DWORD)v67 )
    {
      v37 = (int *)(v66 + 8);
      a2 = (unsigned __int64)v7 + 4 * (unsigned int)v67;
      do
      {
        v38 = *v37;
        ++v36;
        v37 += 4;
        *(v36 - 1) = v38;
      }
      while ( (_DWORD *)a2 != v36 );
    }
  }
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = v34;
  *(_QWORD *)(a1 + 16) = v34;
LABEL_33:
  if ( v18 != v68 )
    _libc_free(v18, a2);
  return a1;
}
