// Function: sub_27BEE10
// Address: 0x27bee10
//
__int64 __fastcall sub_27BEE10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rsi
  int v13; // ebx
  unsigned int i; // eax
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // r14
  _QWORD *v29; // r13
  __int64 v30; // rax
  _QWORD *v31; // r14
  _QWORD *v32; // r13
  __int64 v33; // rax
  __int64 v34; // [rsp+10h] [rbp-160h]
  unsigned __int64 v35; // [rsp+10h] [rbp-160h]
  __int64 v36; // [rsp+18h] [rbp-158h]
  char v37; // [rsp+18h] [rbp-158h]
  _BYTE v38[16]; // [rsp+20h] [rbp-150h] BYREF
  __int64 (__fastcall *v39)(_BYTE *, __int64, int); // [rsp+30h] [rbp-140h]
  __int64 (*v40)(); // [rsp+38h] [rbp-138h]
  __int64 v41; // [rsp+40h] [rbp-130h] BYREF
  _QWORD *v42; // [rsp+48h] [rbp-128h]
  __int64 v43; // [rsp+50h] [rbp-120h]
  __int64 v44; // [rsp+58h] [rbp-118h]
  _QWORD v45[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v46; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 *v47; // [rsp+78h] [rbp-F8h]
  __int64 v48; // [rsp+80h] [rbp-F0h]
  __int64 (*v49)(); // [rsp+88h] [rbp-E8h]
  unsigned __int64 v50[19]; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+128h] [rbp-48h]
  __int64 v52; // [rsp+130h] [rbp-40h]
  __int64 v53; // [rsp+138h] [rbp-38h]

  v7 = sub_B6AC80(*(_QWORD *)(a3 + 40), 153);
  if ( v7 && *(_QWORD *)(v7 + 16) )
  {
    sub_B6AC80(*(_QWORD *)(a3 + 40), 169);
  }
  else
  {
    v17 = sub_B6AC80(*(_QWORD *)(a3 + 40), 169);
    if ( !v17 || !*(_QWORD *)(v17 + 16) )
    {
      *(_BYTE *)(a1 + 76) = 1;
      *(_QWORD *)(a1 + 8) = a1 + 32;
      *(_QWORD *)(a1 + 56) = a1 + 80;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
      return a1;
    }
  }
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v34 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v36 = sub_BC1CD0(a4, &unk_4F8FBC8, a3) + 8;
  v9 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v10 = *(unsigned int *)(a4 + 88);
  v11 = *(_QWORD *)(a4 + 72);
  v12 = v9 + 8;
  if ( !(_DWORD)v10 )
    goto LABEL_53;
  v13 = 1;
  for ( i = (v10 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F8F810 >> 9) ^ ((unsigned int)&unk_4F8F810 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v10 - 1) & v16 )
  {
    v15 = v11 + 24LL * i;
    if ( *(_UNKNOWN **)v15 == &unk_4F8F810 && a3 == *(_QWORD *)(v15 + 8) )
      break;
    if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
      goto LABEL_53;
    v16 = v13 + i;
    ++v13;
  }
  if ( v15 != v11 + 24 * v10 && (v19 = *(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL)) != 0 )
  {
    v20 = *(_QWORD *)(v19 + 8);
    v21 = sub_22077B0(0x2F8u);
    v22 = v21;
    if ( v21 )
    {
      *(_QWORD *)v21 = v20;
      *(_QWORD *)(v21 + 8) = v21 + 24;
      *(_QWORD *)(v21 + 16) = 0x1000000000LL;
      *(_QWORD *)(v21 + 416) = v21 + 440;
      *(_QWORD *)(v21 + 504) = v21 + 520;
      *(_QWORD *)(v21 + 512) = 0x800000000LL;
      *(_QWORD *)(v21 + 408) = 0;
      *(_QWORD *)(v21 + 424) = 8;
      *(_DWORD *)(v21 + 432) = 0;
      *(_BYTE *)(v21 + 436) = 1;
      *(_DWORD *)(v21 + 720) = 0;
      *(_QWORD *)(v21 + 728) = 0;
      *(_QWORD *)(v21 + 736) = v21 + 720;
      *(_QWORD *)(v21 + 744) = v21 + 720;
      *(_QWORD *)(v21 + 752) = 0;
    }
    v40 = sub_27B8DF0;
    v39 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_27B8E00;
    v23 = *(_QWORD *)(v8 + 104);
  }
  else
  {
LABEL_53:
    v22 = 0;
    v40 = sub_27B8DF0;
    v39 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_27B8E00;
    v23 = *(_QWORD *)(v8 + 104);
  }
  v44 = v12;
  v41 = v8 + 8;
  v42 = (_QWORD *)v36;
  v45[0] = v22;
  v43 = v34;
  v45[1] = v23;
  v48 = 0;
  sub_27B8E00(&v46, (__int64)v38, 2);
  v50[0] = v24;
  v35 = v24;
  v49 = v40;
  v50[18] = 0;
  v48 = (__int64)v39;
  v50[1] = 0x1000000000LL;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v37 = sub_27BCF90(&v41);
  sub_C7D6A0(v51, 8LL * (unsigned int)v53, 8);
  if ( v50[0] != v35 )
    _libc_free(v50[0]);
  if ( v48 )
    ((void (__fastcall *)(__int64 *, __int64 *, __int64))v48)(&v46, &v46, 3);
  if ( v39 )
    v39(v38, (__int64)v38, 3);
  v27 = a1 + 80;
  if ( v37 )
  {
    v45[0] = &unk_4F82408;
    v42 = v45;
    v43 = 0x100000002LL;
    LODWORD(v44) = 0;
    BYTE4(v44) = 1;
    v46 = 0;
    v47 = v50;
    v48 = 2;
    LODWORD(v49) = 0;
    BYTE4(v49) = 1;
    v41 = 1;
    sub_27B9A30((__int64)&v41, (__int64)&unk_4F8F810, v25, (__int64)v45, v26, v27);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v45, (__int64)&v41);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v50, (__int64)&v46);
    if ( !BYTE4(v49) )
      _libc_free((unsigned __int64)v47);
    if ( !BYTE4(v44) )
      _libc_free((unsigned __int64)v42);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v27;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  if ( v22 )
  {
    sub_27B9790(*(_QWORD **)(v22 + 728));
    v28 = *(_QWORD **)(v22 + 504);
    v29 = &v28[3 * *(unsigned int *)(v22 + 512)];
    if ( v28 != v29 )
    {
      do
      {
        v30 = *(v29 - 1);
        v29 -= 3;
        if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
          sub_BD60C0(v29);
      }
      while ( v28 != v29 );
      v29 = *(_QWORD **)(v22 + 504);
    }
    if ( v29 != (_QWORD *)(v22 + 520) )
      _libc_free((unsigned __int64)v29);
    if ( !*(_BYTE *)(v22 + 436) )
      _libc_free(*(_QWORD *)(v22 + 416));
    v31 = *(_QWORD **)(v22 + 8);
    v32 = &v31[3 * *(unsigned int *)(v22 + 16)];
    if ( v31 != v32 )
    {
      do
      {
        v33 = *(v32 - 1);
        v32 -= 3;
        if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
          sub_BD60C0(v32);
      }
      while ( v31 != v32 );
      v32 = *(_QWORD **)(v22 + 8);
    }
    if ( v32 != (_QWORD *)(v22 + 24) )
      _libc_free((unsigned __int64)v32);
    j_j___libc_free_0(v22);
  }
  return a1;
}
