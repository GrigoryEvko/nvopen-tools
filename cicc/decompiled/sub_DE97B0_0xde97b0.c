// Function: sub_DE97B0
// Address: 0xde97b0
//
__int64 __fastcall sub_DE97B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v12; // eax
  __int64 v13; // rsi
  int v14; // edi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int i; // eax
  char **v23; // rsi
  char *v24; // r15
  int v25; // eax
  int v26; // eax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // esi
  __int64 v33; // rcx
  __int64 v34; // r9
  __int64 v35; // r10
  __int64 v36; // r8
  __int64 *v37; // r11
  __int64 j; // rdx
  __int64 *v39; // rax
  __int64 v40; // rdi
  int v41; // edx
  __int64 v42; // rdx
  char *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  char *v46; // rdi
  char *v47; // rax
  __int64 *v48; // rax
  char v49; // al
  int v50; // ecx
  __int64 v51; // rdx
  int v52; // ecx
  int v53; // [rsp+8h] [rbp-F8h]
  __int64 *v54; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v55; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v56; // [rsp+28h] [rbp-D8h]
  char *v57; // [rsp+30h] [rbp-D0h]
  __int64 v58; // [rsp+38h] [rbp-C8h]
  char v59; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+60h] [rbp-A0h]
  char *v61[2]; // [rsp+68h] [rbp-98h] BYREF
  _BYTE v62[24]; // [rsp+78h] [rbp-88h] BYREF
  char *v63; // [rsp+90h] [rbp-70h] BYREF
  char *v64; // [rsp+98h] [rbp-68h] BYREF
  __int64 v65; // [rsp+A0h] [rbp-60h]
  _BYTE v66[24]; // [rsp+A8h] [rbp-58h] BYREF
  char v67; // [rsp+C0h] [rbp-40h]

  v7 = a3;
  v8 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) != 12 )
    goto LABEL_13;
  v9 = *(_QWORD *)(a2 + 48);
  v10 = *(_QWORD *)(v8 + 40);
  v12 = *(_DWORD *)(v9 + 24);
  v13 = *(_QWORD *)(v9 + 8);
  if ( !v12 )
    goto LABEL_13;
  v14 = v12 - 1;
  v15 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v16 = (__int64 *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( v10 != *v16 )
  {
    v26 = 1;
    while ( v17 != -4096 )
    {
      a6 = (unsigned int)(v26 + 1);
      v15 = v14 & (v26 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v10 == *v16 )
        goto LABEL_4;
      v26 = a6;
    }
LABEL_13:
    *(_BYTE *)(a1 + 48) = 0;
    return a1;
  }
LABEL_4:
  v18 = v16[1];
  if ( !v18 || v10 != **(_QWORD **)(v18 + 32) )
    goto LABEL_13;
  v19 = *(unsigned int *)(a2 + 1216);
  v20 = *(_QWORD *)(a2 + 1200);
  if ( (_DWORD)v19 )
  {
    v21 = 1;
    for ( i = (v19 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)
                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)))); ; i = (v19 - 1) & v25 )
    {
      v23 = (char **)(v20 + ((unsigned __int64)i << 6));
      v24 = *v23;
      if ( (char *)v7 == *v23 && (char *)v18 == v23[1] )
        break;
      if ( v24 == (char *)-4096LL && v23[1] == (char *)-4096LL )
        goto LABEL_16;
      v25 = v21 + i;
      v21 = (unsigned int)(v21 + 1);
    }
    v42 = v20 + (v19 << 6);
    if ( v23 != (char **)v42 )
    {
      v43 = v23[2];
      v64 = v66;
      v63 = v43;
      v65 = 0x300000000LL;
      v44 = *((unsigned int *)v23 + 8);
      if ( (_DWORD)v44 )
      {
        v23 += 3;
        sub_D915C0((__int64)&v64, (__int64)v23, v42, v44, v21, a6);
        v43 = v63;
      }
      if ( v43 == v24 + 32 )
      {
        *(_BYTE *)(a1 + 48) = 0;
      }
      else
      {
        *(_QWORD *)a1 = v43;
        v45 = (unsigned int)v65;
        *(_QWORD *)(a1 + 8) = a1 + 24;
        *(_QWORD *)(a1 + 16) = 0x300000000LL;
        if ( (_DWORD)v45 )
        {
          v23 = &v64;
          sub_D91460(a1 + 8, &v64, v45, v44, v21, a6);
        }
        *(_BYTE *)(a1 + 48) = 1;
      }
      v46 = v64;
      if ( v64 == v66 )
        return a1;
      goto LABEL_35;
    }
  }
LABEL_16:
  v23 = (char **)a2;
  sub_DE8D20((__int64)&v63, a2, v7);
  if ( !v67 )
  {
    v32 = *(_DWORD *)(a2 + 1216);
    v33 = v7 + 32;
    v55 = v7;
    v57 = &v59;
    v34 = a2 + 1192;
    v58 = 0x300000000LL;
    v60 = v7 + 32;
    v61[0] = v62;
    v61[1] = (char *)0x300000000LL;
    v56 = v18;
    if ( v32 )
    {
      v35 = *(_QWORD *)(a2 + 1200);
      v36 = v32 - 1;
      v37 = 0;
      v53 = 1;
      for ( j = (unsigned int)v36
              & ((unsigned int)((0xBF58476D1CE4E5B9LL
                               * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)
                                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)))); ; j = (unsigned int)v36 & v41 )
      {
        v39 = (__int64 *)(v35 + ((unsigned __int64)(unsigned int)j << 6));
        v40 = *v39;
        if ( v7 == *v39 && v18 == v39[1] )
        {
          v48 = v39 + 2;
          goto LABEL_43;
        }
        if ( v40 == -4096 )
        {
          if ( v39[1] == -4096 )
          {
            v52 = *(_DWORD *)(a2 + 1208);
            if ( v37 )
              v39 = v37;
            ++*(_QWORD *)(a2 + 1192);
            v50 = v52 + 1;
            v54 = v39;
            if ( 4 * v50 < 3 * v32 )
            {
              if ( v32 - *(_DWORD *)(a2 + 1212) - v50 <= v32 >> 3 )
              {
LABEL_53:
                sub_DA6830(a2 + 1192, v32);
                sub_D9E620(a2 + 1192, &v55, &v54);
                v7 = v55;
                v50 = *(_DWORD *)(a2 + 1208) + 1;
                v39 = v54;
              }
              *(_DWORD *)(a2 + 1208) = v50;
              if ( *v39 != -4096 || v39[1] != -4096 )
                --*(_DWORD *)(a2 + 1212);
              *v39 = v7;
              v51 = v56;
              v39[2] = 0;
              v33 = v60;
              v39[1] = v51;
              j = (__int64)(v39 + 5);
              v48 = v39 + 2;
              v48[1] = j;
              v48[2] = 0x300000000LL;
LABEL_43:
              *v48 = v33;
              v23 = v61;
              sub_D91460((__int64)(v48 + 1), v61, j, v33, v36, v34);
              if ( v61[0] != v62 )
                _libc_free(v61[0], v61);
              *(_BYTE *)(a1 + 48) = 0;
              v49 = v67;
LABEL_46:
              if ( v49 )
                goto LABEL_39;
              return a1;
            }
LABEL_52:
            v32 *= 2;
            goto LABEL_53;
          }
        }
        else if ( v40 == -8192 && v39[1] == -8192 && !v37 )
        {
          v37 = (__int64 *)(v35 + ((unsigned __int64)(unsigned int)j << 6));
        }
        v41 = v53 + j;
        ++v53;
      }
    }
    ++*(_QWORD *)(a2 + 1192);
    v54 = 0;
    goto LABEL_52;
  }
  v47 = v63;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)a1 = v47;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x300000000LL;
  if ( (_DWORD)v65 )
  {
    v23 = &v64;
    sub_D91460(a1 + 8, &v64, v28, v29, v30, v31);
    *(_BYTE *)(a1 + 48) = 1;
    v49 = v67;
    goto LABEL_46;
  }
  *(_BYTE *)(a1 + 48) = 1;
LABEL_39:
  v46 = v64;
  if ( v64 != v66 )
LABEL_35:
    _libc_free(v46, v23);
  return a1;
}
