// Function: sub_2E70350
// Address: 0x2e70350
//
void __fastcall sub_2E70350(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r14
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int i; // eax
  char *v29; // rdx
  __int64 v30; // r15
  unsigned int v31; // r10d
  __int64 v32; // rax
  unsigned int v33; // r10d
  unsigned int *v34; // r13
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  char *v38; // r10
  __int64 v39; // r11
  __int64 *v40; // r13
  _QWORD *v41; // rax
  __int64 v42; // r14
  char *v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  _QWORD *v48; // [rsp+0h] [rbp-1500h]
  __int64 v49; // [rsp+8h] [rbp-14F8h]
  unsigned int v51; // [rsp+34h] [rbp-14CCh]
  unsigned int v52; // [rsp+38h] [rbp-14C8h]
  char *v53; // [rsp+38h] [rbp-14C8h]
  __int64 v54; // [rsp+38h] [rbp-14C8h]
  __int64 *v55; // [rsp+40h] [rbp-14C0h] BYREF
  unsigned int v56; // [rsp+48h] [rbp-14B8h]
  char v57; // [rsp+50h] [rbp-14B0h] BYREF
  char *v58; // [rsp+90h] [rbp-1470h] BYREF
  __int64 v59; // [rsp+98h] [rbp-1468h]
  _QWORD v60[128]; // [rsp+A0h] [rbp-1460h] BYREF
  _QWORD v61[2]; // [rsp+4A0h] [rbp-1060h] BYREF
  _QWORD v62[66]; // [rsp+4B0h] [rbp-1050h] BYREF
  char v63; // [rsp+6C0h] [rbp-E40h] BYREF
  __int64 v64; // [rsp+14C0h] [rbp-40h]

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 104);
  sub_2E6DCE0((__int64 *)(a1 + 24));
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 116) = 0;
  *(_QWORD *)(a1 + 104) = v4;
  if ( a2 )
  {
    v3 = *(_QWORD *)(a2 + 16);
    if ( v3 )
    {
      v9 = *(_QWORD *)(a2 + 8);
      if ( v3 != v9 )
        sub_2E6E5B0(*(_QWORD *)(a2 + 8), *(_QWORD *)(a2 + 16));
      if ( v3 + 304 != v9 + 304 )
        sub_2E6E5B0(v9 + 304, v3 + 304);
      *(_BYTE *)(v9 + 608) = *(_BYTE *)(v3 + 608);
      if ( v9 + 616 != v3 + 616 )
      {
        v5 = *(unsigned int *)(v3 + 624);
        v10 = *(unsigned int *)(v9 + 624);
        v11 = *(_DWORD *)(v3 + 624);
        if ( v5 <= v10 )
        {
          if ( *(_DWORD *)(v3 + 624) )
            memmove(*(void **)(v9 + 616), *(const void **)(v3 + 616), 16 * v5);
        }
        else
        {
          if ( v5 > *(unsigned int *)(v9 + 628) )
          {
            *(_DWORD *)(v9 + 624) = 0;
            sub_C8D5F0(v9 + 616, (const void *)(v9 + 632), v5, 0x10u, v7, v8);
            v5 = *(unsigned int *)(v3 + 624);
            v6 = 0;
          }
          else
          {
            v6 = 16 * v10;
            if ( *(_DWORD *)(v9 + 624) )
            {
              v54 = 16 * v10;
              memmove(*(void **)(v9 + 616), *(const void **)(v3 + 616), 16 * v10);
              v5 = *(unsigned int *)(v3 + 624);
              v6 = v54;
            }
          }
          v12 = *(_QWORD *)(v3 + 616);
          v5 *= 16LL;
          if ( v12 + v6 != v5 + v12 )
            memcpy((void *)(v6 + *(_QWORD *)(v9 + 616)), (const void *)(v12 + v6), v5 - v6);
        }
        *(_DWORD *)(v9 + 624) = v11;
      }
      v3 = a2;
      v4 = *(_QWORD *)(a1 + 104);
    }
  }
  v64 = v3;
  v61[0] = v62;
  v61[1] = 0x4000000001LL;
  v62[64] = &v63;
  v62[65] = 0x4000000000LL;
  v59 = 0x100000000LL;
  v58 = (char *)v60;
  v13 = *(_QWORD *)(v4 + 328);
  v62[0] = 0;
  sub_2E6D5A0((__int64)&v58, v13, v5, v6, v7, v8);
  sub_2E6C4A0(a1, &v58, v14, v15, v16, v17);
  if ( v58 != (char *)v60 )
    _libc_free((unsigned __int64)v58);
  v22 = v61;
  v23 = **(_QWORD **)a1;
  v60[1] = 0;
  v58 = (char *)v60;
  v60[0] = v23;
  v59 = 0x4000000001LL;
  v51 = 0;
  *(_DWORD *)(sub_2E6F1C0((__int64)v61, v23, v18, v19, v20, v21) + 4) = 0;
  for ( i = v59; (_DWORD)v59; i = v59 )
  {
    while ( 1 )
    {
      v29 = &v58[16 * i - 16];
      v30 = *(_QWORD *)v29;
      v31 = *((_DWORD *)v29 + 2);
      LODWORD(v59) = i - 1;
      v23 = v30;
      v52 = v31;
      v32 = sub_2E6F1C0((__int64)v22, v30, (__int64)v29, (__int64)v58, v26, v27);
      v33 = v52;
      v34 = (unsigned int *)v32;
      v35 = *(unsigned int *)(v32 + 32);
      v25 = v34[9];
      if ( v35 + 1 > v25 )
      {
        v23 = (__int64)(v34 + 10);
        sub_C8D5F0((__int64)(v34 + 6), v34 + 10, v35 + 1, 4u, v26, v27);
        v35 = v34[8];
        v33 = v52;
      }
      *(_DWORD *)(*((_QWORD *)v34 + 3) + 4 * v35) = v33;
      v24 = *v34;
      ++v34[8];
      if ( !(_DWORD)v24 )
      {
        ++v51;
        v34[1] = v33;
        v34[3] = v51;
        v34[2] = v51;
        *v34 = v51;
        sub_2E6D5A0((__int64)v22, v30, v24, v25, v26, v27);
        v23 = v30;
        sub_2E6EC80(&v55, v30, v64, v36, v37);
        v24 = v56;
        v38 = (char *)&v55[v56];
        if ( v55 != (__int64 *)v38 )
        {
          v24 = (unsigned int)v59;
          v39 = v51;
          v40 = v55;
          v41 = v22;
          do
          {
            v25 = HIDWORD(v59);
            v26 = v24 + 1;
            v42 = *v40;
            v2 = v39 | v2 & 0xFFFFFFFF00000000LL;
            if ( v24 + 1 > (unsigned __int64)HIDWORD(v59) )
            {
              v48 = v41;
              v49 = v39;
              v53 = v38;
              sub_C8D5F0((__int64)&v58, v60, v24 + 1, 0x10u, v26, v27);
              v24 = (unsigned int)v59;
              v41 = v48;
              v39 = v49;
              v38 = v53;
            }
            v43 = &v58[16 * v24];
            ++v40;
            *(_QWORD *)v43 = v42;
            *((_QWORD *)v43 + 1) = v2;
            v23 = (unsigned int)v59;
            v24 = (unsigned int)(v59 + 1);
            LODWORD(v59) = v59 + 1;
          }
          while ( v38 != (char *)v40 );
          v38 = (char *)v55;
          v22 = v41;
        }
        if ( v38 != &v57 )
          break;
      }
      i = v59;
      if ( !(_DWORD)v59 )
        goto LABEL_32;
    }
    _libc_free((unsigned __int64)v38);
  }
LABEL_32:
  if ( v58 != (char *)v60 )
    _libc_free((unsigned __int64)v58);
  sub_2E6F370((__int64)v22, v23, v24, v25, v26, v27);
  if ( a2 )
    *(_BYTE *)a2 = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v44 = (__int64 *)sub_2E6E2F0(a1, **(_QWORD **)a1, 0);
    *(_QWORD *)(a1 + 96) = v44;
    sub_2E6FB50(v22, a1, *v44, v45, v46, v47);
  }
  sub_2E6D840((__int64)v22);
}
