// Function: sub_1D96690
// Address: 0x1d96690
//
void __fastcall sub_1D96690(__int64 a1, char *a2, __int64 a3, char a4)
{
  __int64 v5; // r12
  _QWORD *v7; // r15
  __int64 *v8; // r13
  __int64 v9; // rax
  _QWORD *v10; // r10
  __int64 *v11; // r9
  __int64 *v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int64 *v16; // rcx
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 v19; // rdi
  unsigned __int64 *v20; // r9
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  _BYTE *v24; // r11
  _BYTE *v25; // r9
  __int64 *v26; // r8
  int v27; // eax
  size_t v28; // r10
  __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 *v32; // r14
  __int64 v33; // rsi
  unsigned int v34; // r12d
  _QWORD *v35; // rax
  unsigned int v36; // r8d
  const void *v37; // r15
  unsigned __int64 v38; // r13
  size_t v39; // r14
  __int64 v40; // rdx
  __int64 *v41; // rdi
  int v42; // eax
  char v43; // dl
  char v44; // dl
  char v45; // al
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  size_t v48; // [rsp+10h] [rbp-A0h]
  __int64 *v49; // [rsp+18h] [rbp-98h]
  _QWORD *v50; // [rsp+18h] [rbp-98h]
  __int64 *v51; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v52; // [rsp+20h] [rbp-90h]
  unsigned int v53; // [rsp+20h] [rbp-90h]
  _BYTE *v54; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v57; // [rsp+38h] [rbp-78h]
  __int64 *v58; // [rsp+38h] [rbp-78h]
  _BYTE *v59; // [rsp+38h] [rbp-78h]
  __int64 *v60; // [rsp+38h] [rbp-78h]
  __int64 v61; // [rsp+48h] [rbp-68h] BYREF
  __int64 *v62; // [rsp+50h] [rbp-60h] BYREF
  __int64 v63; // [rsp+58h] [rbp-58h]
  _BYTE dest[80]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a3;
  v7 = *(_QWORD **)(a3 + 16);
  v8 = (__int64 *)sub_1DD5EE0(v7);
  v9 = sub_1DD5EE0(*((_QWORD *)a2 + 2));
  v10 = v7 + 2;
  v11 = (__int64 *)v9;
  v12 = (__int64 *)v7[4];
  if ( v8 != v12 && v11 != v8 )
  {
    v13 = *((_QWORD *)a2 + 2) + 16LL;
    if ( (_QWORD *)v13 != v10 )
    {
      v49 = v11;
      v51 = (__int64 *)v7[4];
      sub_1DD5C00(v13, v7 + 2, v51, v8);
      v11 = v49;
      v12 = v51;
      v10 = v7 + 2;
    }
    if ( v8 != v11 && v8 != v12 )
    {
      v14 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v8;
      *v8 = *v8 & 7 | *v12 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = *v11;
      *(_QWORD *)(v14 + 8) = v11;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *v12 = v15 | *v12 & 7;
      *(_QWORD *)(v15 + 8) = v12;
      *v11 = v14 | *v11 & 7;
    }
  }
  v16 = v7 + 3;
  if ( v8 != v7 + 3 )
  {
    v17 = *(_QWORD *)(a1 + 544);
    v18 = *(__int64 (**)())(*(_QWORD *)v17 + 656LL);
    if ( v18 == sub_1D918C0
      || (v50 = v10,
          v60 = v11,
          v45 = ((__int64 (__fastcall *)(__int64, __int64 *))v18)(v17, v8),
          v20 = (unsigned __int64 *)v60,
          v16 = v7 + 3,
          v10 = v50,
          !v45) )
    {
      v19 = *((_QWORD *)a2 + 2);
      v20 = (unsigned __int64 *)(v19 + 24);
    }
    else
    {
      v19 = *((_QWORD *)a2 + 2);
    }
    v21 = (_QWORD *)(v19 + 16);
    if ( v20 != v16 )
    {
      if ( v10 != v21 )
      {
        v52 = v20;
        v57 = v16;
        sub_1DD5C00(v21, v10, v8, v16);
        v20 = v52;
        v16 = v57;
      }
      if ( v16 != v20 && v16 != (unsigned __int64 *)v8 )
      {
        v22 = v7[3] & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v16;
        v7[3] = v7[3] & 7LL | *v8 & 0xFFFFFFFFFFFFFFF8LL;
        v23 = *v20;
        *(_QWORD *)(v22 + 8) = v20;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *v8 = v23 | *v8 & 7;
        *(_QWORD *)(v23 + 8) = v8;
        *v20 = v22 | *v20 & 7;
      }
    }
  }
  if ( (*a2 & 0x10) != 0 )
    sub_1D96570(*(unsigned int **)(*((_QWORD *)a2 + 2) + 112LL), *(unsigned int **)(*((_QWORD *)a2 + 2) + 120LL));
  v24 = (_BYTE *)v7[12];
  v25 = (_BYTE *)v7[11];
  v26 = (__int64 *)dest;
  v63 = 0x400000000LL;
  v27 = 0;
  v28 = v24 - v25;
  v62 = (__int64 *)dest;
  v29 = (v24 - v25) >> 3;
  if ( (unsigned __int64)(v24 - v25) > 0x20 )
  {
    v59 = v25;
    v48 = v24 - v25;
    v54 = v24;
    sub_16CD150((__int64)&v62, dest, (v24 - v25) >> 3, 8, (int)dest, (int)v25);
    v27 = v63;
    v28 = v48;
    v24 = v54;
    v25 = v59;
    v26 = &v62[(unsigned int)v63];
  }
  if ( v25 != v24 )
  {
    memmove(v26, v25, v28);
    v27 = v63;
  }
  v53 = 0;
  LODWORD(v63) = v29 + v27;
  v30 = (unsigned int)(v29 + v27);
  v31 = v7[1];
  if ( v31 == v7[7] + 320LL )
    v31 = 0;
  if ( (*(_BYTE *)v5 & 0x40) == 0 )
    v31 = 0;
  if ( a4 )
  {
    if ( (unsigned __int8)sub_1DD6970(*((_QWORD *)a2 + 2), v7) )
    {
      v53 = sub_1DF1780(*(_QWORD *)(a1 + 560), *((_QWORD *)a2 + 2), v7);
      sub_1DD91B0(*((_QWORD *)a2 + 2), v7, 0);
    }
    v30 = (unsigned int)v63;
  }
  v32 = v62;
  v58 = &v62[v30];
  if ( v62 != v58 )
  {
    v46 = v5;
    while ( 1 )
    {
      v33 = *v32;
      v61 = v33;
      if ( v33 == v31 )
        goto LABEL_31;
      if ( a4 )
      {
        v34 = sub_1DF1780(*(_QWORD *)(a1 + 560), v7, v33);
        if ( v53 )
          v34 = (v53 * (unsigned __int64)v34 + 0x40000000) >> 31;
        sub_1DD91B0(v7, v61, 0);
        if ( !(unsigned __int8)sub_1DD6970(*((_QWORD *)a2 + 2), v61) )
        {
          sub_1DD8FE0(*((_QWORD *)a2 + 2), v61, v34);
          goto LABEL_31;
        }
        v47 = *((_QWORD *)a2 + 2);
        sub_1DF1780(*(_QWORD *)(a1 + 560), v47, v61);
        ++v32;
        v35 = sub_1D91D40(*(_QWORD **)(*((_QWORD *)a2 + 2) + 88LL), *(_QWORD *)(*((_QWORD *)a2 + 2) + 96LL), &v61);
        sub_1DD76A0(v47, v35, v36);
        if ( v58 == v32 )
        {
LABEL_38:
          v5 = v46;
          break;
        }
      }
      else
      {
        sub_1DD91B0(v7, v33, 0);
LABEL_31:
        if ( v58 == ++v32 )
          goto LABEL_38;
      }
    }
  }
  if ( v7 != (_QWORD *)(*(_QWORD *)(v7[7] + 320LL) & 0xFFFFFFFFFFFFFFF8LL) )
    sub_1DD6900(v7);
  if ( (*a2 & 0x10) != 0 && (*(_BYTE *)v5 & 0x10) != 0 )
    sub_1D96570(*(unsigned int **)(*((_QWORD *)a2 + 2) + 112LL), *(unsigned int **)(*((_QWORD *)a2 + 2) + 120LL));
  v37 = *(const void **)(v5 + 216);
  v38 = *(unsigned int *)(v5 + 224);
  v39 = 40 * v38;
  v40 = *((unsigned int *)a2 + 56);
  if ( v38 > (unsigned __int64)*((unsigned int *)a2 + 57) - v40 )
  {
    sub_16CD150((__int64)(a2 + 216), a2 + 232, v38 + v40, 40, (int)v26, (int)v25);
    v40 = *((unsigned int *)a2 + 56);
  }
  if ( v39 )
  {
    memcpy((void *)(*((_QWORD *)a2 + 27) + 40 * v40), v37, v39);
    LODWORD(v40) = *((_DWORD *)a2 + 56);
  }
  v41 = v62;
  *((_DWORD *)a2 + 56) = v38 + v40;
  v42 = *(_DWORD *)(v5 + 4);
  *(_DWORD *)(v5 + 224) = 0;
  *((_DWORD *)a2 + 1) += v42;
  *((_DWORD *)a2 + 2) += *(_DWORD *)(v5 + 8);
  *((_DWORD *)a2 + 3) += *(_DWORD *)(v5 + 12);
  v43 = *(_BYTE *)(v5 + 1);
  *(_DWORD *)(v5 + 4) = 0;
  *(_QWORD *)(v5 + 8) = 0;
  LOBYTE(v42) = (a2[1] | v43) & 2 | a2[1] & 0xFD;
  v44 = *a2;
  a2[1] = v42;
  *a2 = v44 & 0xBB | *(_BYTE *)v5 & 0x40;
  *(_BYTE *)v5 &= ~4u;
  if ( v41 != (__int64 *)dest )
    _libc_free((unsigned __int64)v41);
}
