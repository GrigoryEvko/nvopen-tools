// Function: sub_FEAA90
// Address: 0xfeaa90
//
char __fastcall sub_FEAA90(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _DWORD *v5; // r14
  unsigned int v7; // ebx
  __int64 v8; // rdx
  __int64 v9; // r10
  unsigned int *v10; // r9
  __int64 v11; // rax
  _DWORD *v12; // rax
  __int64 v13; // r10
  unsigned __int64 v14; // rdx
  _BYTE *v15; // r12
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rbx
  unsigned int v18; // r15d
  __int64 v19; // rax
  _DWORD *v20; // rdi
  __int64 v21; // r11
  unsigned __int64 *v22; // rdx
  __int64 v23; // r14
  unsigned int v24; // edx
  unsigned __int64 v25; // rax
  bool v26; // cf
  unsigned __int64 v27; // rcx
  _BYTE *v28; // rdx
  __int64 *v29; // r9
  unsigned __int64 *v30; // rdx
  _DWORD *v31; // rdi
  __int64 *v33; // [rsp+10h] [rbp-D0h]
  __int64 *v34; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v35; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v36; // [rsp+18h] [rbp-C8h]
  _BYTE *v37; // [rsp+20h] [rbp-C0h]
  __int64 v38; // [rsp+20h] [rbp-C0h]
  _BYTE *v39; // [rsp+38h] [rbp-A8h]
  unsigned int v40; // [rsp+48h] [rbp-98h] BYREF
  unsigned int v41; // [rsp+4Ch] [rbp-94h] BYREF
  _BYTE *v42; // [rsp+50h] [rbp-90h] BYREF
  __int64 v43; // [rsp+58h] [rbp-88h]
  _BYTE v44[64]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-40h]
  char v46; // [rsp+A8h] [rbp-38h]

  v5 = a2;
  v7 = 0;
  v8 = (unsigned int)a2[3];
  v42 = v44;
  v43 = 0x400000000LL;
  v45 = 0;
  v46 = 0;
  if ( (_DWORD)v8 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *((_QWORD *)v5 + 12);
        v10 = (unsigned int *)(v9 + 4LL * v7);
        v11 = 0;
        if ( (_DWORD)v8 != 1 )
        {
          a2 = (_DWORD *)(v9 + 4 * v8);
          v12 = sub_FE8370(*((_DWORD **)v5 + 12), (__int64)a2, (_DWORD *)(v9 + 4LL * v7));
          v11 = 8 * (((__int64)v12 - v13) >> 2);
        }
        v14 = *(_QWORD *)(*((_QWORD *)v5 + 16) + v11);
        if ( v14 )
          break;
        v8 = (unsigned int)v5[3];
        if ( (unsigned int)v8 <= ++v7 )
          goto LABEL_8;
      }
      a2 = v10;
      ++v7;
      sub_FE8630((__int64)&v42, v10, v14, 0, a5, (__int64)v10);
      v8 = (unsigned int)v5[3];
    }
    while ( (unsigned int)v8 > v7 );
  }
LABEL_8:
  sub_FE9FC0((__int64)&v42);
  v15 = v42;
  v16 = (unsigned __int64)&v42[16 * (unsigned int)v43];
  v39 = (_BYTE *)v16;
  if ( (_BYTE *)v16 == v42 )
    goto LABEL_23;
  v17 = -1;
  v18 = v45;
  do
  {
    while ( 1 )
    {
      v23 = *((_QWORD *)v15 + 1);
      v24 = v18;
      v18 -= v23;
      sub_F02DB0(&v41, v23, v24);
      a2 = (_DWORD *)v17;
      v40 = v41;
      v25 = sub_F02E20(&v40, v17);
      v26 = v17 < v25;
      v17 -= v25;
      v27 = v25;
      if ( v26 )
        v17 = 0;
      v16 = *(_QWORD *)(a1 + 64);
      v28 = (_BYTE *)(v16 + 24LL * *((unsigned int *)v15 + 1));
      v29 = (__int64 *)*((_QWORD *)v28 + 1);
      if ( !v29 )
        goto LABEL_20;
      v19 = *((unsigned int *)v29 + 3);
      v20 = (_DWORD *)v29[12];
      if ( (unsigned int)v19 > 1 )
        break;
      LODWORD(v16) = *v20;
      if ( *(_DWORD *)v28 == *v20 )
        goto LABEL_12;
LABEL_20:
      v30 = (unsigned __int64 *)(v28 + 16);
LABEL_21:
      *v30 = v27;
      v15 += 16;
      if ( v15 == v39 )
        goto LABEL_22;
    }
    a2 = &v20[v19];
    v33 = (__int64 *)*((_QWORD *)v28 + 1);
    v35 = v27;
    v37 = v28;
    LOBYTE(v16) = sub_FDC990(v20, a2, v28);
    v28 = v37;
    v27 = v35;
    v29 = v33;
    if ( !(_BYTE)v16 )
    {
      v30 = (unsigned __int64 *)(v37 + 16);
      goto LABEL_21;
    }
LABEL_12:
    if ( !*((_BYTE *)v29 + 8) )
      goto LABEL_20;
    v21 = *v29;
    if ( !*v29 )
      goto LABEL_15;
    v16 = *(unsigned int *)(v21 + 12);
    if ( (unsigned int)v16 <= 1 )
      goto LABEL_15;
    v31 = *(_DWORD **)(v21 + 96);
    v34 = v29;
    v36 = v27;
    a2 = &v31[v16];
    v38 = *v29;
    LOBYTE(v16) = sub_FDC990(v31, a2, v28);
    v27 = v36;
    v29 = v34;
    if ( !(_BYTE)v16 || (v22 = (unsigned __int64 *)(v38 + 152), !*(_BYTE *)(v38 + 8)) )
LABEL_15:
      v22 = (unsigned __int64 *)(v29 + 19);
    *v22 = v27;
    v15 += 16;
  }
  while ( v15 != v39 );
LABEL_22:
  v15 = v42;
LABEL_23:
  if ( v15 != v44 )
    LOBYTE(v16) = _libc_free(v15, a2);
  return v16;
}
