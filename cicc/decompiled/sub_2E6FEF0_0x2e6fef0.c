// Function: sub_2E6FEF0
// Address: 0x2e6fef0
//
void __fastcall sub_2E6FEF0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _QWORD *v8; // rsi
  __int64 v10; // r14
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r13
  int v18; // r10d
  __int64 v19; // rax
  int v20; // r10d
  __int64 v21; // rbx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 *v26; // rbx
  char *v27; // r11
  __int64 v28; // r10
  __int64 *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // r11
  _QWORD *v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _BYTE *v41; // rbx
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // [rsp+8h] [rbp-1508h]
  __int64 v45; // [rsp+8h] [rbp-1508h]
  __int64 v46; // [rsp+10h] [rbp-1500h]
  __int64 v47; // [rsp+10h] [rbp-1500h]
  unsigned int v49; // [rsp+38h] [rbp-14D8h]
  unsigned int v50; // [rsp+38h] [rbp-14D8h]
  unsigned int v51; // [rsp+3Ch] [rbp-14D4h]
  int v52; // [rsp+48h] [rbp-14C8h]
  unsigned __int64 v53; // [rsp+48h] [rbp-14C8h]
  __int64 v54; // [rsp+48h] [rbp-14C8h]
  char *v55; // [rsp+50h] [rbp-14C0h] BYREF
  int v56; // [rsp+58h] [rbp-14B8h]
  char v57; // [rsp+60h] [rbp-14B0h] BYREF
  _QWORD *v58; // [rsp+A0h] [rbp-1470h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-1468h]
  _QWORD v60[128]; // [rsp+B0h] [rbp-1460h] BYREF
  unsigned __int64 v61[2]; // [rsp+4B0h] [rbp-1060h] BYREF
  _QWORD v62[64]; // [rsp+4C0h] [rbp-1050h] BYREF
  _BYTE *v63; // [rsp+6C0h] [rbp-E50h]
  __int64 v64; // [rsp+6C8h] [rbp-E48h]
  _BYTE v65[3584]; // [rsp+6D0h] [rbp-E40h] BYREF
  __int64 v66; // [rsp+14D0h] [rbp-40h]

  v8 = (_QWORD *)a3;
  v10 = a1;
  v63 = v65;
  v61[0] = (unsigned __int64)v62;
  v64 = 0x4000000000LL;
  v61[1] = 0x4000000001LL;
  v62[0] = 0;
  v66 = a2;
  v58 = v60;
  v60[0] = a3;
  v60[1] = 0;
  v59 = 0x4000000001LL;
  v51 = 0;
  *(_DWORD *)(sub_2E6F1C0((__int64)v61, a3, a3, (__int64)v60, a5, a6) + 4) = 0;
  v15 = v59;
  if ( (_DWORD)v59 )
  {
    while ( 1 )
    {
LABEL_4:
      v16 = (__int64)&v58[2 * v15 - 2];
      v17 = *(_QWORD *)v16;
      v18 = *(_DWORD *)(v16 + 8);
      LODWORD(v59) = v15 - 1;
      v8 = (_QWORD *)v17;
      v52 = v18;
      v19 = sub_2E6F1C0((__int64)v61, v17, v16, (__int64)v58, v13, v14);
      v20 = v52;
      v21 = v19;
      v22 = *(unsigned int *)(v19 + 32);
      v12 = *(unsigned int *)(v21 + 36);
      if ( v22 + 1 > v12 )
      {
        v8 = (_QWORD *)(v21 + 40);
        sub_C8D5F0(v21 + 24, (const void *)(v21 + 40), v22 + 1, 4u, v13, v14);
        v22 = *(unsigned int *)(v21 + 32);
        v20 = v52;
      }
      v11 = *(_QWORD *)(v21 + 24);
      *(_DWORD *)(v11 + 4 * v22) = v20;
      v23 = *(_DWORD *)v21;
      ++*(_DWORD *)(v21 + 32);
      if ( !v23 )
        break;
LABEL_3:
      v15 = v59;
      if ( !(_DWORD)v59 )
        goto LABEL_22;
    }
    *(_DWORD *)(v21 + 4) = v20;
    *(_DWORD *)(v21 + 12) = ++v51;
    *(_DWORD *)(v21 + 8) = v51;
    *(_DWORD *)v21 = v51;
    sub_2E6D5A0((__int64)v61, v17, v11, v12, v13, v14);
    v8 = (_QWORD *)v17;
    sub_2E6EC80(&v55, v17, v66, v24, v25);
    v26 = (__int64 *)v55;
    v27 = &v55[8 * v56];
    if ( v55 == v27 )
      goto LABEL_20;
    v13 = v6;
    v14 = v51;
    v28 = v17;
    v29 = (__int64 *)&v55[8 * v56];
    while ( 1 )
    {
      v34 = *v26;
      if ( *v26 )
      {
        v30 = (unsigned int)(*(_DWORD *)(v34 + 24) + 1);
        if ( (unsigned int)(*(_DWORD *)(v34 + 24) + 1) < *(_DWORD *)(a1 + 32) )
          goto LABEL_10;
LABEL_16:
        v12 = HIDWORD(v59);
        v35 = (unsigned int)v59;
        v36 = v13 & 0xFFFFFFFF00000000LL | (unsigned int)v14;
        v11 = (unsigned int)v59 + 1LL;
        v13 = v36;
        if ( v11 > HIDWORD(v59) )
        {
          v8 = v60;
          v44 = v36;
          v46 = v28;
          v49 = v14;
          v53 = v36;
          sub_C8D5F0((__int64)&v58, v60, v11, 0x10u, v36, v14);
          v35 = (unsigned int)v59;
          v13 = v44;
          v28 = v46;
          v14 = v49;
          v36 = v53;
        }
        ++v26;
        v37 = &v58[2 * v35];
        *v37 = v34;
        v37[1] = v36;
        LODWORD(v59) = v59 + 1;
        if ( v29 == v26 )
          goto LABEL_19;
      }
      else
      {
        v30 = 0;
        if ( !*(_DWORD *)(a1 + 32) )
          goto LABEL_16;
LABEL_10:
        v31 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v30);
        if ( !v31 )
          goto LABEL_16;
        v32 = *(unsigned int *)(a5 + 8);
        v12 = *(unsigned int *)(a5 + 12);
        v11 = v32 + 1;
        if ( v32 + 1 > v12 )
        {
          v8 = (_QWORD *)(a5 + 16);
          v45 = v13;
          v47 = v28;
          v50 = v14;
          v54 = v31;
          sub_C8D5F0(a5, (const void *)(a5 + 16), v11, 0x10u, v13, v14);
          v32 = *(unsigned int *)(a5 + 8);
          v13 = v45;
          v28 = v47;
          v14 = v50;
          v31 = v54;
        }
        ++v26;
        v33 = (__int64 *)(*(_QWORD *)a5 + 16 * v32);
        *v33 = v28;
        v33[1] = v31;
        ++*(_DWORD *)(a5 + 8);
        if ( v29 == v26 )
        {
LABEL_19:
          v27 = v55;
          v6 = v13;
LABEL_20:
          if ( v27 == &v57 )
            goto LABEL_3;
          _libc_free((unsigned __int64)v27);
          v15 = v59;
          if ( !(_DWORD)v59 )
          {
LABEL_22:
            v10 = a1;
            break;
          }
          goto LABEL_4;
        }
      }
    }
  }
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  sub_2E6F370((__int64)v61, (__int64)v8, v11, v12, v13, v14);
  sub_2E6FB50(v61, v10, *a4, v38, v39, v40);
  v41 = v63;
  v42 = (unsigned __int64)&v63[56 * (unsigned int)v64];
  if ( v63 != (_BYTE *)v42 )
  {
    do
    {
      v42 -= 56LL;
      v43 = *(_QWORD *)(v42 + 24);
      if ( v43 != v42 + 40 )
        _libc_free(v43);
    }
    while ( v41 != (_BYTE *)v42 );
    v42 = (unsigned __int64)v63;
  }
  if ( (_BYTE *)v42 != v65 )
    _libc_free(v42);
  if ( (_QWORD *)v61[0] != v62 )
    _libc_free(v61[0]);
}
