// Function: sub_2E71BA0
// Address: 0x2e71ba0
//
void __fastcall sub_2E71BA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v7; // rcx
  unsigned int v8; // eax
  unsigned int v9; // r13d
  __int64 v10; // r14
  __int64 *v11; // rbx
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  unsigned int v18; // eax
  void *v19; // r10
  size_t v20; // r13
  unsigned __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // r9
  char v25; // dl
  __int64 v26; // rsi
  __int64 v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r13
  __int64 *v34; // rbx
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // r12
  __int64 *v38; // r15
  __int64 *v39; // r14
  unsigned int v40; // r13d
  unsigned int v41; // eax
  __int64 v42; // rdx
  _BYTE *v43; // rdi
  _BYTE *v44; // rdi
  __int64 v45; // rax
  int v46; // r10d
  void *src; // [rsp+0h] [rbp-B0h]
  __int64 *v48; // [rsp+8h] [rbp-A8h]
  __int64 *v49; // [rsp+8h] [rbp-A8h]
  void *v50; // [rsp+8h] [rbp-A8h]
  int v51; // [rsp+8h] [rbp-A8h]
  void *v52; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+10h] [rbp-A0h]
  __int64 *v54; // [rsp+10h] [rbp-A0h]
  _BYTE *v55; // [rsp+10h] [rbp-A0h]
  int v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  int v58; // [rsp+10h] [rbp-A0h]
  int v59; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+20h] [rbp-90h]
  _BYTE *v62; // [rsp+30h] [rbp-80h] BYREF
  __int64 v63; // [rsp+38h] [rbp-78h]
  _BYTE v64[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = a1;
  if ( a3 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
    v8 = *(_DWORD *)(a3 + 24) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = *(_DWORD *)(a1 + 32);
  if ( v8 >= v9 )
    return;
  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(__int64 **)(v10 + 8 * v7);
  if ( !v11 )
    return;
  if ( a4 )
  {
    v12 = (unsigned int)(*(_DWORD *)(a4 + 24) + 1);
    v13 = *(_DWORD *)(a4 + 24) + 1;
  }
  else
  {
    v12 = 0;
    v13 = 0;
  }
  if ( v9 <= v13 )
    return;
  v14 = *(__int64 **)(v10 + 8 * v12);
  if ( !v14 )
    return;
  v15 = sub_2E6D000(a1, a3, a4);
  if ( v15 )
  {
    v17 = (unsigned int)(*(_DWORD *)(v15 + 24) + 1);
    v18 = *(_DWORD *)(v15 + 24) + 1;
  }
  else
  {
    v17 = 0;
    v18 = 0;
  }
  if ( v9 > v18 && v14 == *(__int64 **)(v10 + 8 * v17) )
    return;
  *(_BYTE *)(a1 + 112) = 0;
  if ( v11 != (__int64 *)v14[1] )
  {
LABEL_14:
    sub_2E71750(v4, a2, *v11, *v14);
    return;
  }
  v19 = *(void **)(*v14 + 64);
  v60 = *v14;
  v20 = 8LL * *(unsigned int *)(*v14 + 72);
  v21 = *(unsigned int *)(*v14 + 72);
  if ( a2 )
  {
    v22 = *(_QWORD *)(a2 + 8);
    v62 = v64;
    v63 = 0x800000000LL;
    if ( v20 > 0x40 )
    {
      src = v19;
      v50 = (void *)v22;
      v56 = v21;
      sub_C8D5F0((__int64)&v62, v64, v21, 8u, v21, v22);
      LODWORD(v21) = v56;
      v22 = (__int64)v50;
      v19 = src;
      v43 = &v62[8 * (unsigned int)v63];
    }
    else
    {
      if ( !v20 )
        goto LABEL_21;
      v43 = v64;
    }
    v51 = v21;
    v57 = v22;
    memcpy(v43, v19, v20);
    LODWORD(v20) = v63;
    LODWORD(v21) = v51;
    v22 = v57;
LABEL_21:
    v53 = v22;
    LODWORD(v63) = v21 + v20;
    sub_2E6E9F0((__int64)&v62);
    v24 = v53;
    v25 = *(_BYTE *)(v53 + 312) & 1;
    if ( v25 )
    {
      v26 = v53 + 320;
      v27 = 3;
    }
    else
    {
      v27 = *(unsigned int *)(v53 + 328);
      v26 = *(_QWORD *)(v53 + 320);
      if ( !(_DWORD)v27 )
        goto LABEL_59;
      v27 = (unsigned int)(v27 - 1);
    }
    v28 = v27 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
    v54 = (__int64 *)(v26 + 72LL * v28);
    v29 = *v54;
    if ( v60 == *v54 )
    {
LABEL_24:
      v30 = 288;
      if ( !v25 )
        v30 = 72LL * *(unsigned int *)(v24 + 328);
      if ( v54 != (__int64 *)(v26 + v30) )
      {
        v31 = v54[1];
        v32 = *((unsigned int *)v54 + 4);
        if ( v31 != v31 + 8 * v32 )
        {
          v48 = v11;
          v33 = (__int64 *)(v31 + 8 * v32);
          v34 = (__int64 *)v54[1];
          do
          {
            v35 = *v34++;
            sub_2E6EB60((__int64)&v62, v35);
          }
          while ( v33 != v34 );
          v11 = v48;
        }
        sub_2E6EC00((__int64)&v62, (__int64)(v54 + 5), v32, v27, v23, v24);
      }
      goto LABEL_32;
    }
    v46 = 1;
    while ( v29 != -4096 )
    {
      v23 = (unsigned int)(v46 + 1);
      v28 = v27 & (v46 + v28);
      v54 = (__int64 *)(v26 + 72LL * v28);
      v29 = *v54;
      if ( v60 == *v54 )
        goto LABEL_24;
      ++v46;
    }
    if ( v25 )
    {
      v45 = 288;
      goto LABEL_60;
    }
    v27 = *(unsigned int *)(v24 + 328);
LABEL_59:
    v45 = 72 * v27;
LABEL_60:
    v54 = (__int64 *)(v26 + v45);
    goto LABEL_24;
  }
  v62 = v64;
  v63 = 0x800000000LL;
  if ( v20 > 0x40 )
  {
    v52 = v19;
    v58 = v21;
    sub_C8D5F0((__int64)&v62, v64, v21, 8u, v21, v16);
    LODWORD(v21) = v58;
    v19 = v52;
    v44 = &v62[8 * (unsigned int)v63];
LABEL_57:
    v59 = v21;
    memcpy(v44, v19, v20);
    LODWORD(v20) = v63;
    LODWORD(v21) = v59;
    goto LABEL_44;
  }
  if ( v20 )
  {
    v44 = v64;
    goto LABEL_57;
  }
LABEL_44:
  LODWORD(v63) = v21 + v20;
  sub_2E6E9F0((__int64)&v62);
LABEL_32:
  v55 = v62;
  if ( v62 != &v62[8 * (unsigned int)v63] )
  {
    v36 = *(_DWORD *)(v4 + 32);
    v49 = v14;
    v37 = v4;
    v38 = (__int64 *)v62;
    v39 = (__int64 *)&v62[8 * (unsigned int)v63];
    v40 = v36;
    while ( 1 )
    {
      v42 = *v38;
      if ( *v38 )
      {
        v27 = (unsigned int)(*(_DWORD *)(v42 + 24) + 1);
        v41 = *(_DWORD *)(v42 + 24) + 1;
      }
      else
      {
        v27 = 0;
        v41 = 0;
      }
      if ( v41 < v40 && *(_QWORD *)(*(_QWORD *)(v37 + 24) + 8 * v27) && v60 != sub_2E6D000(v37, v60, v42) )
        break;
      if ( v39 == ++v38 )
      {
        v4 = v37;
        v14 = v49;
        goto LABEL_46;
      }
    }
    v4 = v37;
    v14 = v49;
    if ( v55 != v64 )
      _libc_free((unsigned __int64)v55);
    goto LABEL_14;
  }
LABEL_46:
  if ( v55 != v64 )
    _libc_free((unsigned __int64)v55);
  sub_2E70D60(v4, a2, (__int64)v14, v27, v23, v24);
}
