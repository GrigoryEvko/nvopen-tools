// Function: sub_B1EAC0
// Address: 0xb1eac0
//
__int64 __fastcall sub_B1EAC0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  unsigned __int64 v5; // r12
  _BYTE *v9; // rsi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r10
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // r13
  __int64 v17; // rax
  int v18; // eax
  __int64 *v19; // rbx
  char *v20; // r11
  unsigned __int64 v21; // r8
  unsigned int v22; // r9d
  __int64 *v23; // r12
  __int64 v24; // r10
  __int64 v25; // rdx
  __int64 v26; // r11
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  unsigned __int64 v31; // r11
  unsigned __int64 v32; // rdx
  _QWORD *v33; // rax
  _BYTE *v34; // rbx
  __int64 result; // rax
  _BYTE *v36; // r12
  _BYTE *v37; // rdi
  unsigned __int64 v38; // [rsp+8h] [rbp-1518h]
  unsigned __int64 v39; // [rsp+8h] [rbp-1518h]
  __int64 v40; // [rsp+10h] [rbp-1510h]
  __int64 v41; // [rsp+10h] [rbp-1510h]
  unsigned int v43; // [rsp+48h] [rbp-14D8h]
  unsigned int v44; // [rsp+48h] [rbp-14D8h]
  unsigned int v45; // [rsp+4Ch] [rbp-14D4h]
  __int64 v46; // [rsp+58h] [rbp-14C8h]
  __int64 v47; // [rsp+58h] [rbp-14C8h]
  unsigned __int64 v48; // [rsp+58h] [rbp-14C8h]
  __int64 v49; // [rsp+58h] [rbp-14C8h]
  char *v50; // [rsp+60h] [rbp-14C0h] BYREF
  int v51; // [rsp+68h] [rbp-14B8h]
  char v52; // [rsp+70h] [rbp-14B0h] BYREF
  _BYTE *v53; // [rsp+B0h] [rbp-1470h] BYREF
  unsigned int v54; // [rsp+B8h] [rbp-1468h]
  unsigned int v55; // [rsp+BCh] [rbp-1464h]
  _BYTE v56[1024]; // [rsp+C0h] [rbp-1460h] BYREF
  _QWORD v57[2]; // [rsp+4C0h] [rbp-1060h] BYREF
  _QWORD v58[64]; // [rsp+4D0h] [rbp-1050h] BYREF
  _BYTE *v59; // [rsp+6D0h] [rbp-E50h]
  __int64 v60; // [rsp+6D8h] [rbp-E48h]
  _BYTE v61[3584]; // [rsp+6E0h] [rbp-E40h] BYREF
  __int64 v62; // [rsp+14E0h] [rbp-40h]

  v57[0] = v58;
  v57[1] = 0x4000000001LL;
  v59 = v61;
  v60 = 0x4000000000LL;
  v62 = a2;
  v50 = (char *)a3;
  v58[0] = 0;
  v51 = 0;
  sub_B1C510(&v53, &v50, 1);
  v9 = v56;
  v45 = 0;
  *(_DWORD *)(sub_B1E0B0((__int64)v57, a3) + 4) = 0;
  v10 = v54;
  if ( v54 )
  {
    while ( 1 )
    {
LABEL_4:
      v11 = (__int64)&v53[16 * v10 - 16];
      v12 = *(_QWORD *)v11;
      v13 = *(_DWORD *)(v11 + 8);
      v54 = v10 - 1;
      v9 = (_BYTE *)v12;
      v46 = v12;
      v14 = sub_B1E0B0((__int64)v57, v12);
      v15 = v46;
      v16 = v14;
      v17 = *(unsigned int *)(v14 + 32);
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 36) )
      {
        v9 = (_BYTE *)(v16 + 40);
        sub_C8D5F0(v16 + 24, v16 + 40, v17 + 1, 4);
        v17 = *(unsigned int *)(v16 + 32);
        v15 = v46;
      }
      *(_DWORD *)(*(_QWORD *)(v16 + 24) + 4 * v17) = v13;
      v18 = *(_DWORD *)v16;
      ++*(_DWORD *)(v16 + 32);
      if ( !v18 )
        break;
LABEL_3:
      v10 = v54;
      if ( !v54 )
        goto LABEL_22;
    }
    *(_DWORD *)(v16 + 4) = v13;
    ++v45;
    v47 = v15;
    *(_DWORD *)(v16 + 12) = v45;
    *(_DWORD *)(v16 + 8) = v45;
    *(_DWORD *)v16 = v45;
    sub_B1A4E0((__int64)v57, v15);
    v9 = (_BYTE *)v47;
    sub_B1D150(&v50, v47, v62);
    v19 = (__int64 *)v50;
    v20 = &v50[8 * v51];
    if ( v50 == v20 )
      goto LABEL_20;
    v21 = v5;
    v22 = v45;
    v23 = (__int64 *)&v50[8 * v51];
    v24 = v47;
    while ( 1 )
    {
      v29 = *v19;
      if ( *v19 )
      {
        v25 = (unsigned int)(*(_DWORD *)(v29 + 44) + 1);
        if ( (unsigned int)(*(_DWORD *)(v29 + 44) + 1) < *(_DWORD *)(a1 + 32) )
          goto LABEL_10;
LABEL_16:
        v30 = v54;
        v31 = v21 & 0xFFFFFFFF00000000LL | v22;
        v32 = v54 + 1LL;
        v21 = v31;
        if ( v32 > v55 )
        {
          v9 = v56;
          v38 = v31;
          v40 = v24;
          v43 = v22;
          v48 = v31;
          sub_C8D5F0(&v53, v56, v32, 16);
          v30 = v54;
          v21 = v38;
          v24 = v40;
          v22 = v43;
          v31 = v48;
        }
        ++v19;
        v33 = &v53[16 * v30];
        *v33 = v29;
        v33[1] = v31;
        ++v54;
        if ( v23 == v19 )
          goto LABEL_19;
      }
      else
      {
        v25 = 0;
        if ( !*(_DWORD *)(a1 + 32) )
          goto LABEL_16;
LABEL_10:
        v26 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v25);
        if ( !v26 )
          goto LABEL_16;
        v27 = *(unsigned int *)(a5 + 8);
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          v9 = (_BYTE *)(a5 + 16);
          v39 = v21;
          v41 = v24;
          v44 = v22;
          v49 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v25);
          sub_C8D5F0(a5, a5 + 16, v27 + 1, 16);
          v27 = *(unsigned int *)(a5 + 8);
          v21 = v39;
          v24 = v41;
          v22 = v44;
          v26 = v49;
        }
        ++v19;
        v28 = (__int64 *)(*(_QWORD *)a5 + 16 * v27);
        *v28 = v24;
        v28[1] = v26;
        ++*(_DWORD *)(a5 + 8);
        if ( v23 == v19 )
        {
LABEL_19:
          v20 = v50;
          v5 = v21;
LABEL_20:
          if ( v20 == &v52 )
            goto LABEL_3;
          _libc_free(v20, v9);
          v10 = v54;
          if ( !v54 )
            break;
          goto LABEL_4;
        }
      }
    }
  }
LABEL_22:
  if ( v53 != v56 )
    _libc_free(v53, v9);
  sub_B1E260((__int64)v57);
  sub_B1E720(v57, a1, *a4);
  v34 = v59;
  result = 7LL * (unsigned int)v60;
  v36 = &v59[56 * (unsigned int)v60];
  if ( v59 != v36 )
  {
    do
    {
      v36 -= 56;
      v37 = (_BYTE *)*((_QWORD *)v36 + 3);
      result = (__int64)(v36 + 40);
      if ( v37 != v36 + 40 )
        result = _libc_free(v37, a1);
    }
    while ( v34 != v36 );
    v36 = v59;
  }
  if ( v36 != v61 )
    result = _libc_free(v36, a1);
  if ( (_QWORD *)v57[0] != v58 )
    return _libc_free(v57[0], a1);
  return result;
}
