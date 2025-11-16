// Function: sub_2EB6550
// Address: 0x2eb6550
//
void __fastcall sub_2EB6550(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
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
  __int64 v26; // r9
  __int64 *v27; // rbx
  char *v28; // r11
  __int64 v29; // r10
  __int64 *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // r11
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  unsigned __int64 v37; // r11
  _QWORD *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  _BYTE *v42; // rbx
  unsigned __int64 v43; // r12
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // [rsp+8h] [rbp-1508h]
  __int64 v46; // [rsp+8h] [rbp-1508h]
  __int64 v47; // [rsp+10h] [rbp-1500h]
  __int64 v48; // [rsp+10h] [rbp-1500h]
  unsigned int v50; // [rsp+38h] [rbp-14D8h]
  unsigned int v51; // [rsp+38h] [rbp-14D8h]
  unsigned int v52; // [rsp+3Ch] [rbp-14D4h]
  int v53; // [rsp+48h] [rbp-14C8h]
  unsigned __int64 v54; // [rsp+48h] [rbp-14C8h]
  __int64 v55; // [rsp+48h] [rbp-14C8h]
  char *v56; // [rsp+50h] [rbp-14C0h] BYREF
  int v57; // [rsp+58h] [rbp-14B8h]
  char v58; // [rsp+60h] [rbp-14B0h] BYREF
  _QWORD *v59; // [rsp+A0h] [rbp-1470h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-1468h]
  _QWORD v61[128]; // [rsp+B0h] [rbp-1460h] BYREF
  unsigned __int64 v62[2]; // [rsp+4B0h] [rbp-1060h] BYREF
  _QWORD v63[64]; // [rsp+4C0h] [rbp-1050h] BYREF
  _BYTE *v64; // [rsp+6C0h] [rbp-E50h]
  __int64 v65; // [rsp+6C8h] [rbp-E48h]
  _BYTE v66[3584]; // [rsp+6D0h] [rbp-E40h] BYREF
  __int64 v67; // [rsp+14D0h] [rbp-40h]

  v8 = (_QWORD *)a3;
  v10 = a1;
  v64 = v66;
  v62[0] = (unsigned __int64)v63;
  v65 = 0x4000000000LL;
  v62[1] = 0x4000000001LL;
  v63[0] = 0;
  v67 = a2;
  v59 = v61;
  v61[0] = a3;
  v61[1] = 0;
  v60 = 0x4000000001LL;
  v52 = 0;
  *(_DWORD *)(sub_2EB5B40((__int64)v62, a3, a3, (__int64)v61, a5, a6) + 4) = 0;
  v15 = v60;
  if ( (_DWORD)v60 )
  {
    while ( 1 )
    {
LABEL_4:
      v16 = (__int64)&v59[2 * v15 - 2];
      v17 = *(_QWORD *)v16;
      v18 = *(_DWORD *)(v16 + 8);
      LODWORD(v60) = v15 - 1;
      v8 = (_QWORD *)v17;
      v53 = v18;
      v19 = sub_2EB5B40((__int64)v62, v17, v16, (__int64)v59, v13, v14);
      v20 = v53;
      v21 = v19;
      v22 = *(unsigned int *)(v19 + 32);
      v12 = *(unsigned int *)(v21 + 36);
      if ( v22 + 1 > v12 )
      {
        v8 = (_QWORD *)(v21 + 40);
        sub_C8D5F0(v21 + 24, (const void *)(v21 + 40), v22 + 1, 4u, v13, v14);
        v22 = *(unsigned int *)(v21 + 32);
        v20 = v53;
      }
      v11 = *(_QWORD *)(v21 + 24);
      *(_DWORD *)(v11 + 4 * v22) = v20;
      v23 = *(_DWORD *)v21;
      ++*(_DWORD *)(v21 + 32);
      if ( !v23 )
        break;
LABEL_3:
      v15 = v60;
      if ( !(_DWORD)v60 )
        goto LABEL_22;
    }
    *(_DWORD *)(v21 + 4) = v20;
    *(_DWORD *)(v21 + 12) = ++v52;
    *(_DWORD *)(v21 + 8) = v52;
    *(_DWORD *)v21 = v52;
    sub_2E6D5A0((__int64)v62, v17, v11, v12, v13, v14);
    v8 = (_QWORD *)v17;
    sub_2EB52F0(&v56, v17, v67, v24, v25, v26);
    v27 = (__int64 *)v56;
    v28 = &v56[8 * v57];
    if ( v56 == v28 )
      goto LABEL_20;
    v13 = v6;
    v14 = v52;
    v29 = v17;
    v30 = (__int64 *)&v56[8 * v57];
    while ( 1 )
    {
      v35 = *v27;
      if ( *v27 )
      {
        v31 = (unsigned int)(*(_DWORD *)(v35 + 24) + 1);
        if ( (unsigned int)(*(_DWORD *)(v35 + 24) + 1) < *(_DWORD *)(a1 + 56) )
          goto LABEL_10;
LABEL_16:
        v12 = HIDWORD(v60);
        v36 = (unsigned int)v60;
        v37 = v13 & 0xFFFFFFFF00000000LL | (unsigned int)v14;
        v11 = (unsigned int)v60 + 1LL;
        v13 = v37;
        if ( v11 > HIDWORD(v60) )
        {
          v8 = v61;
          v45 = v37;
          v47 = v29;
          v50 = v14;
          v54 = v37;
          sub_C8D5F0((__int64)&v59, v61, v11, 0x10u, v37, v14);
          v36 = (unsigned int)v60;
          v13 = v45;
          v29 = v47;
          v14 = v50;
          v37 = v54;
        }
        ++v27;
        v38 = &v59[2 * v36];
        *v38 = v35;
        v38[1] = v37;
        LODWORD(v60) = v60 + 1;
        if ( v30 == v27 )
          goto LABEL_19;
      }
      else
      {
        v31 = 0;
        if ( !*(_DWORD *)(a1 + 56) )
          goto LABEL_16;
LABEL_10:
        v32 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v31);
        if ( !v32 )
          goto LABEL_16;
        v33 = *(unsigned int *)(a5 + 8);
        v12 = *(unsigned int *)(a5 + 12);
        v11 = v33 + 1;
        if ( v33 + 1 > v12 )
        {
          v8 = (_QWORD *)(a5 + 16);
          v46 = v13;
          v48 = v29;
          v51 = v14;
          v55 = v32;
          sub_C8D5F0(a5, (const void *)(a5 + 16), v11, 0x10u, v13, v14);
          v33 = *(unsigned int *)(a5 + 8);
          v13 = v46;
          v29 = v48;
          v14 = v51;
          v32 = v55;
        }
        ++v27;
        v34 = (__int64 *)(*(_QWORD *)a5 + 16 * v33);
        *v34 = v29;
        v34[1] = v32;
        ++*(_DWORD *)(a5 + 8);
        if ( v30 == v27 )
        {
LABEL_19:
          v28 = v56;
          v6 = v13;
LABEL_20:
          if ( v28 == &v58 )
            goto LABEL_3;
          _libc_free((unsigned __int64)v28);
          v15 = v60;
          if ( !(_DWORD)v60 )
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
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  sub_2EB5CF0((__int64)v62, (__int64)v8, v11, v12, v13, v14);
  sub_2EB61B0(v62, v10, *a4, v39, v40, v41);
  v42 = v64;
  v43 = (unsigned __int64)&v64[56 * (unsigned int)v65];
  if ( v64 != (_BYTE *)v43 )
  {
    do
    {
      v43 -= 56LL;
      v44 = *(_QWORD *)(v43 + 24);
      if ( v44 != v43 + 40 )
        _libc_free(v44);
    }
    while ( v42 != (_BYTE *)v43 );
    v43 = (unsigned __int64)v64;
  }
  if ( (_BYTE *)v43 != v66 )
    _libc_free(v43);
  if ( (_QWORD *)v62[0] != v63 )
    _libc_free(v62[0]);
}
