// Function: sub_2D749D0
// Address: 0x2d749d0
//
__int64 __fastcall sub_2D749D0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 v12; // r12
  _BYTE *v14; // r13
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rsi
  int v22; // edx
  unsigned __int64 *v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdi
  _QWORD *v27; // r15
  _QWORD *v28; // r14
  __int64 v29; // rax
  _QWORD *v30; // r15
  _QWORD *v31; // r12
  __int64 v32; // rax
  int v33; // r10d
  int v34; // eax
  __int64 v35; // r8
  char *v36; // r15
  unsigned __int64 v37; // [rsp+20h] [rbp-890h] BYREF
  __int64 v38; // [rsp+28h] [rbp-888h]
  __int64 v39; // [rsp+30h] [rbp-880h]
  int v40; // [rsp+38h] [rbp-878h]
  _BYTE *v41; // [rsp+40h] [rbp-870h] BYREF
  __int64 v42; // [rsp+48h] [rbp-868h]
  _BYTE v43[1024]; // [rsp+50h] [rbp-860h] BYREF
  unsigned __int64 v44; // [rsp+450h] [rbp-460h] BYREF
  __int64 v45; // [rsp+458h] [rbp-458h]
  __int64 v46; // [rsp+460h] [rbp-450h]
  _BYTE *v47; // [rsp+468h] [rbp-448h] BYREF
  __int64 v48; // [rsp+470h] [rbp-440h]
  _BYTE v49[1080]; // [rsp+478h] [rbp-438h] BYREF

  v4 = a2[2];
  v44 = 0;
  v45 = 0;
  v46 = v4;
  if ( v4 == 0 || v4 == -4096 || v4 == -8192 )
  {
    v37 = 0;
    v38 = 0;
    v39 = v4;
LABEL_15:
    v40 = 0;
    goto LABEL_16;
  }
  sub_BD6050(&v44, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  LODWORD(v47) = 0;
  v37 = 0;
  v39 = v46;
  v38 = 0;
  if ( v46 == 0 || v46 == -4096 || v46 == -8192 )
    goto LABEL_15;
  sub_BD6050(&v37, v44 & 0xFFFFFFFFFFFFFFF8LL);
  v40 = (int)v47;
  if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
  {
    sub_BD60C0(&v44);
    v5 = *(_DWORD *)(a1 + 24);
    if ( v5 )
      goto LABEL_8;
LABEL_17:
    ++*(_QWORD *)a1;
    v41 = 0;
LABEL_18:
    sub_28EF240(a1, 2 * v5);
    goto LABEL_19;
  }
LABEL_16:
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
    goto LABEL_17;
LABEL_8:
  v6 = v39;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (v5 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
  v9 = v7 + 32LL * v8;
  v10 = *(_QWORD *)(v9 + 16);
  if ( v39 == v10 )
  {
LABEL_9:
    v11 = *(_DWORD *)(v9 + 24);
    goto LABEL_10;
  }
  v33 = 1;
  v14 = 0;
  while ( v10 != -4096 )
  {
    if ( !v14 && v10 == -8192 )
      v14 = (_BYTE *)v9;
    v8 = (v5 - 1) & (v33 + v8);
    v9 = v7 + 32LL * v8;
    v10 = *(_QWORD *)(v9 + 16);
    if ( v39 == v10 )
      goto LABEL_9;
    ++v33;
  }
  if ( !v14 )
    v14 = (_BYTE *)v9;
  v34 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v34 + 1;
  v41 = v14;
  if ( 4 * (v34 + 1) >= 3 * v5 )
    goto LABEL_18;
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 > v5 >> 3 )
    goto LABEL_20;
  sub_28EF240(a1, v5);
LABEL_19:
  sub_28EE370(a1, (__int64)&v37, &v41);
  v14 = v41;
  v15 = *(_DWORD *)(a1 + 16) + 1;
LABEL_20:
  *(_DWORD *)(a1 + 16) = v15;
  v44 = 0;
  v45 = 0;
  v46 = -4096;
  if ( *((_QWORD *)v14 + 2) != -4096 )
    --*(_DWORD *)(a1 + 20);
  sub_D68D70(&v44);
  sub_2D57220(v14, v39);
  *((_DWORD *)v14 + 6) = v40;
  v41 = v43;
  v42 = 0x2000000000LL;
  sub_D68CD0(&v44, 0, a2);
  v48 = 0x2000000000LL;
  v47 = v49;
  if ( (_DWORD)v42 )
    sub_2D68580((__int64)&v47, (__int64)&v41, (unsigned int)v42, v16);
  v18 = *(unsigned int *)(a1 + 40);
  v19 = &v44;
  v20 = *(_QWORD *)(a1 + 32);
  v21 = v18 + 1;
  v22 = v18;
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v35 = a1 + 32;
    if ( v20 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v20 + 1064 * v18 )
    {
      v19 = &v44;
      sub_2D6DD00(a1 + 32, v21, v18, v16, v35, v17);
      v18 = *(unsigned int *)(a1 + 40);
      v20 = *(_QWORD *)(a1 + 32);
      v22 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      v36 = (char *)&v44 - v20;
      sub_2D6DD00(a1 + 32, v21, v18, v16, v35, v17);
      v20 = *(_QWORD *)(a1 + 32);
      v18 = *(unsigned int *)(a1 + 40);
      v19 = (unsigned __int64 *)&v36[v20];
      v22 = *(_DWORD *)(a1 + 40);
    }
  }
  v23 = (unsigned __int64 *)(1064 * v18 + v20);
  if ( v23 )
  {
    sub_D68CD0(v23, 0, v19);
    v23[3] = (unsigned __int64)(v23 + 5);
    v23[4] = 0x2000000000LL;
    if ( *((_DWORD *)v19 + 8) )
      sub_2D68580((__int64)(v23 + 3), (__int64)(v19 + 3), v24, v25);
    v22 = *(_DWORD *)(a1 + 40);
  }
  v26 = (unsigned int)v48;
  v27 = v47;
  *(_DWORD *)(a1 + 40) = v22 + 1;
  v26 *= 32;
  v28 = (_QWORD *)((char *)v27 + v26);
  if ( v27 != (_QWORD *)((char *)v27 + v26) )
  {
    do
    {
      v29 = *(v28 - 2);
      v28 -= 4;
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0(v28);
    }
    while ( v27 != v28 );
    v28 = v47;
  }
  if ( v28 != (_QWORD *)v49 )
    _libc_free((unsigned __int64)v28);
  sub_D68D70(&v44);
  v30 = v41;
  v31 = &v41[32 * (unsigned int)v42];
  if ( v41 != (_BYTE *)v31 )
  {
    do
    {
      v32 = *(v31 - 2);
      v31 -= 4;
      if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
        sub_BD60C0(v31);
    }
    while ( v30 != v31 );
    v31 = v41;
  }
  if ( v31 != (_QWORD *)v43 )
    _libc_free((unsigned __int64)v31);
  v6 = v39;
  v11 = *(_DWORD *)(a1 + 40) - 1;
  *((_DWORD *)v14 + 6) = v11;
LABEL_10:
  v12 = *(_QWORD *)(a1 + 32) + 1064LL * v11 + 24;
  if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
    sub_BD60C0(&v37);
  return v12;
}
