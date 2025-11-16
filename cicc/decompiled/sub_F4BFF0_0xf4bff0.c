// Function: sub_F4BFF0
// Address: 0xf4bff0
//
__int64 __fastcall sub_F4BFF0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  _QWORD *v4; // r15
  __int64 v6; // rbx
  _BYTE *v7; // r8
  __int64 v8; // r12
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rax
  const char *v14; // rax
  unsigned __int64 v15; // r14
  const char *v16; // rax
  char v17; // bl
  __int64 v18; // rax
  char v19; // bl
  __int64 v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned __int8 *v25; // rbx
  __int64 v26; // rcx
  __int64 v27; // r14
  __int64 v28; // r12
  unsigned __int8 *v29; // r15
  __int64 v30; // rdi
  unsigned int v31; // esi
  __int64 v32; // rdx
  __int64 v33; // r10
  __int64 v34; // rax
  const char *v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  unsigned __int8 *v38; // rax
  int v40; // edx
  int v41; // r10d
  int v42; // edx
  int v43; // ecx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // [rsp+10h] [rbp-C0h]
  __int64 v50; // [rsp+20h] [rbp-B0h]
  const void *v51; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE *v52; // [rsp+38h] [rbp-98h]
  _BYTE *v53; // [rsp+40h] [rbp-90h]
  const char *v54; // [rsp+50h] [rbp-80h] BYREF
  __int64 v55; // [rsp+58h] [rbp-78h]
  _BYTE v56[16]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v57; // [rsp+70h] [rbp-60h]

  v4 = (_QWORD *)a1;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1, a2, (__int64)a3, a4);
    v6 = *(_QWORD *)(a1 + 96);
    v8 = v6 + 40LL * *(_QWORD *)(a1 + 104);
    if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a1, a2, v44, v45);
      v6 = *(_QWORD *)(a1 + 96);
    }
    v7 = v52;
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 96);
    v7 = 0;
    v8 = v6 + 40LL * *(_QWORD *)(a1 + 104);
  }
  if ( v8 != v6 )
  {
    while ( 1 )
    {
      v13 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v13 )
        goto LABEL_9;
      v9 = *(_QWORD *)(a2 + 8);
      v10 = (v13 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v11 = v9 + ((unsigned __int64)v10 << 6);
      v12 = *(_QWORD *)(v11 + 24);
      if ( v12 == v6 )
      {
LABEL_6:
        if ( v11 == v9 + (v13 << 6) )
          goto LABEL_9;
LABEL_7:
        v6 += 40;
        if ( v8 == v6 )
          break;
      }
      else
      {
        v40 = 1;
        while ( v12 != -4096 )
        {
          v41 = v40 + 1;
          v10 = (v13 - 1) & (v40 + v10);
          v11 = v9 + ((unsigned __int64)v10 << 6);
          v12 = *(_QWORD *)(v11 + 24);
          if ( v12 == v6 )
            goto LABEL_6;
          v40 = v41;
        }
LABEL_9:
        v14 = *(const char **)(v6 + 8);
        v54 = v14;
        if ( v53 == v7 )
        {
          sub_9183A0((__int64)&v51, v7, &v54);
          v7 = v52;
          goto LABEL_7;
        }
        if ( v7 )
        {
          *(_QWORD *)v7 = v14;
          v7 = v52;
        }
        v7 += 8;
        v6 += 40;
        v52 = v7;
        if ( v8 == v6 )
          break;
      }
    }
  }
  v15 = sub_BCF480(
          **(__int64 ***)(*(_QWORD *)(a1 + 24) + 16LL),
          v51,
          (v7 - (_BYTE *)v51) >> 3,
          *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8 != 0);
  v47 = *(_QWORD *)(a1 + 40);
  v16 = sub_BD5D20(a1);
  v17 = *(_BYTE *)(a1 + 32);
  v54 = v16;
  v18 = *(_QWORD *)(a1 + 8);
  v57 = 261;
  v19 = v17 & 0xF;
  v55 = v20;
  v21 = *(_DWORD *)(v18 + 8) >> 8;
  v50 = sub_BD2DA0(136);
  if ( v50 )
    sub_B2C3B0(v50, v15, v19, v21, (__int64)&v54, v47);
  v22 = *(unsigned __int8 *)(a1 + 128);
  sub_B2B9F0(v50, v22);
  if ( (*(_BYTE *)(v50 + 2) & 1) != 0 )
    sub_B2C6D0(v50, v22, v23, v24);
  v25 = *(unsigned __int8 **)(v50 + 96);
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1, v22, v23, v24);
    v26 = *(_QWORD *)(a1 + 96);
    v27 = v26 + 40LL * *(_QWORD *)(a1 + 104);
    if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a1, v22, v46, v26);
      v26 = *(_QWORD *)(a1 + 96);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 96);
    v27 = v26 + 40LL * *(_QWORD *)(a1 + 104);
  }
  v28 = v26;
  if ( v26 != v27 )
  {
    v29 = v25;
    while ( 1 )
    {
      while ( 1 )
      {
        v34 = *(unsigned int *)(a2 + 24);
        if ( !(_DWORD)v34 )
          goto LABEL_25;
        v30 = *(_QWORD *)(a2 + 8);
        v31 = (v34 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v32 = v30 + ((unsigned __int64)v31 << 6);
        v33 = *(_QWORD *)(v32 + 24);
        if ( v28 == v33 )
          break;
        v42 = 1;
        while ( v33 != -4096 )
        {
          v43 = v42 + 1;
          v31 = (v34 - 1) & (v42 + v31);
          v32 = v30 + ((unsigned __int64)v31 << 6);
          v33 = *(_QWORD *)(v32 + 24);
          if ( v28 == v33 )
            goto LABEL_22;
          v42 = v43;
        }
LABEL_25:
        v35 = sub_BD5D20(v28);
        v57 = 261;
        v54 = v35;
        v55 = v36;
        sub_BD6B50(v29, &v54);
        v37 = sub_F46C80(a2, v28);
        v38 = (unsigned __int8 *)v37[2];
        if ( v38 == v29 )
          goto LABEL_38;
        if ( v38 != 0 && v38 + 4096 != 0 && v38 != (unsigned __int8 *)-8192LL )
          sub_BD60C0(v37);
        v37[2] = v29;
        if ( v29 == 0 || v29 + 4096 == 0 || v29 == (unsigned __int8 *)-8192LL )
        {
LABEL_38:
          v29 += 40;
          goto LABEL_23;
        }
        sub_BD73F0((__int64)v37);
        v28 += 40;
        v29 += 40;
        if ( v27 == v28 )
        {
LABEL_32:
          v4 = (_QWORD *)a1;
          goto LABEL_33;
        }
      }
LABEL_22:
      if ( v32 == v30 + (v34 << 6) )
        goto LABEL_25;
LABEL_23:
      v28 += 40;
      if ( v27 == v28 )
        goto LABEL_32;
    }
  }
LABEL_33:
  v54 = v56;
  v55 = 0x800000000LL;
  sub_F4BB00(v50, v4, a2, 0, (__int64)&v54, byte_3F871B3, a3, 0, 0);
  if ( v54 != v56 )
    _libc_free(v54, v4);
  if ( v51 )
    j_j___libc_free_0(v51, v53 - (_BYTE *)v51);
  return v50;
}
