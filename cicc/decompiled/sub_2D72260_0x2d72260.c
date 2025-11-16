// Function: sub_2D72260
// Address: 0x2d72260
//
__int64 __fastcall sub_2D72260(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rbx
  unsigned __int64 v13; // r8
  int v14; // eax
  unsigned __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 *v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 *v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rcx
  int *v27; // r12
  int *v28; // rbx
  char v29; // r13
  __int64 v30; // rdi
  _QWORD *v31; // rbx
  _QWORD *v32; // r12
  __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 *v35; // r12
  __int64 v36; // rax
  __int64 v38; // rdi
  char *v39; // rbx
  unsigned __int8 v41; // [rsp+38h] [rbp-3D8h]
  unsigned __int64 v42[4]; // [rsp+40h] [rbp-3D0h] BYREF
  _BYTE v43[48]; // [rsp+60h] [rbp-3B0h] BYREF
  __int64 *v44; // [rsp+90h] [rbp-380h] BYREF
  __int64 v45; // [rsp+98h] [rbp-378h]
  _BYTE v46[384]; // [rsp+A0h] [rbp-370h] BYREF
  __int64 v47; // [rsp+220h] [rbp-1F0h] BYREF
  __int64 v48; // [rsp+228h] [rbp-1E8h]
  _QWORD v49[49]; // [rsp+230h] [rbp-1E0h] BYREF
  int v50; // [rsp+3B8h] [rbp-58h] BYREF
  _QWORD *v51; // [rsp+3C0h] [rbp-50h]
  int *v52; // [rsp+3C8h] [rbp-48h]
  int *v53; // [rsp+3D0h] [rbp-40h]
  __int64 v54; // [rsp+3D8h] [rbp-38h]

  v6 = a2 + 9;
  v8 = a2[10];
  v44 = (__int64 *)v46;
  v45 = 0x1000000000LL;
  v9 = *(__int64 **)(v8 + 8);
  if ( a2 + 9 == v9 )
  {
    v28 = (int *)v49;
    v48 = 0x1000000000LL;
    v47 = (__int64)v49;
    v27 = (int *)v49;
    v50 = 0;
    v51 = 0;
    v52 = &v50;
    v53 = &v50;
    v54 = 0;
    v41 = 0;
  }
  else
  {
    do
    {
      v10 = v9 - 3;
      v47 = 6;
      if ( !v9 )
        v10 = 0;
      v48 = 0;
      v49[0] = v10;
      if ( v10 != 0 && v10 + 512 != 0 && v10 != (__int64 *)-8192LL )
        sub_BD73F0((__int64)&v47);
      v11 = (unsigned int)v45;
      v12 = &v47;
      a2 = v44;
      v13 = (unsigned int)v45 + 1LL;
      v14 = v45;
      if ( v13 > HIDWORD(v45) )
      {
        if ( v44 > &v47 || &v47 >= &v44[3 * (unsigned int)v45] )
        {
          sub_F39130((__int64)&v44, (unsigned int)v45 + 1LL, (unsigned int)v45, a4, v13, a6);
          v11 = (unsigned int)v45;
          a2 = v44;
          v14 = v45;
        }
        else
        {
          v39 = (char *)((char *)&v47 - (char *)v44);
          sub_F39130((__int64)&v44, (unsigned int)v45 + 1LL, (unsigned int)v45, a4, v13, a6);
          a2 = v44;
          v11 = (unsigned int)v45;
          v12 = (__int64 *)&v39[(_QWORD)v44];
          v14 = v45;
        }
      }
      v15 = (unsigned __int64 *)&a2[3 * v11];
      if ( v15 )
      {
        *v15 = 6;
        v16 = v12[2];
        v15[1] = 0;
        v15[2] = v16;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        {
          a2 = (__int64 *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
          sub_BD6050(v15, (unsigned __int64)a2);
        }
        v14 = v45;
      }
      LODWORD(v45) = v14 + 1;
      LOBYTE(a2) = v49[0] != 0;
      if ( v49[0] != -4096 && v49[0] != 0 && v49[0] != -8192 )
        sub_BD60C0(&v47);
      v9 = (__int64 *)v9[1];
    }
    while ( v9 != v6 );
    v50 = 0;
    v51 = 0;
    v54 = 0;
    v47 = (__int64)v49;
    v48 = 0x1000000000LL;
    v52 = &v50;
    v53 = &v50;
    if ( v44 != &v44[3 * (unsigned int)v45] )
    {
      v41 = 0;
      v17 = &v44[3 * (unsigned int)v45];
      v18 = v44;
      while ( 1 )
      {
        v19 = v18[2];
        if ( !v19 )
          goto LABEL_36;
        v20 = sub_AA54C0(v18[2]);
        v21 = (__int64 *)v20;
        if ( !v20 )
          goto LABEL_36;
        if ( v19 == v20 )
          goto LABEL_36;
        if ( (*(_WORD *)(v19 + 2) & 0x7FFF) != 0 )
          goto LABEL_36;
        if ( a3 )
        {
          v22 = (unsigned int)(*(_DWORD *)(v19 + 44) + 1);
          if ( (unsigned int)v22 >= *(_DWORD *)(a3 + 32) || !*(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v22) )
            goto LABEL_36;
        }
        v23 = v21[6] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 *)v23 == v21 + 6 )
          goto LABEL_77;
        if ( !v23 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 > 0xA )
LABEL_77:
          BUG();
        if ( *(_BYTE *)(v23 - 24) != 31 || (*(_DWORD *)(v23 - 20) & 0x7FFFFFF) == 3 )
          goto LABEL_36;
        sub_F39690(v19, 0, *(_QWORD *)(a1 + 56), 0, 0, 0, a3);
        v42[2] = (unsigned __int64)v21;
        v42[0] = 6;
        v42[1] = 0;
        if ( v21 != (__int64 *)-4096LL && v21 != (__int64 *)-8192LL )
          sub_BD73F0((__int64)v42);
        a2 = &v47;
        sub_2D72010((__int64)v43, (__int64)&v47, v42, v24, (__int64)v43, v25);
        sub_D68D70(v42);
        v41 = *(_BYTE *)(a1 + 832);
        if ( v41 )
        {
          v18 += 3;
          sub_D695C0((__int64)v43, a1 + 840, v21, v26, (__int64)v43, a1 + 840);
          a2 = (__int64 *)v19;
          sub_25DDDB0(a1 + 840, v19);
          if ( v18 == v17 )
          {
LABEL_37:
            if ( v54 )
            {
              v27 = v52;
              v28 = &v50;
              v29 = 0;
              goto LABEL_39;
            }
            v27 = (int *)v47;
            v28 = (int *)(v47 + 24LL * (unsigned int)v48);
            goto LABEL_73;
          }
        }
        else
        {
          v41 = 1;
LABEL_36:
          v18 += 3;
          if ( v18 == v17 )
            goto LABEL_37;
        }
      }
    }
    v27 = (int *)v49;
    v41 = 0;
    v28 = (int *)v49;
  }
LABEL_73:
  v29 = 1;
LABEL_39:
  while ( v29 )
  {
    if ( v27 == v28 )
      goto LABEL_45;
    v38 = *((_QWORD *)v27 + 2);
    if ( v38 )
      sub_F3F2F0(v38, (__int64)a2);
    v27 += 6;
  }
  while ( v27 != v28 )
  {
    v30 = *((_QWORD *)v27 + 6);
    if ( v30 )
      sub_F3F2F0(v30, (__int64)a2);
    v27 = (int *)sub_220EF30((__int64)v27);
  }
LABEL_45:
  sub_2D58B20(v51);
  v31 = (_QWORD *)v47;
  v32 = (_QWORD *)(v47 + 24LL * (unsigned int)v48);
  if ( (_QWORD *)v47 != v32 )
  {
    do
    {
      v33 = *(v32 - 1);
      v32 -= 3;
      if ( v33 != -4096 && v33 != 0 && v33 != -8192 )
        sub_BD60C0(v32);
    }
    while ( v31 != v32 );
    v32 = (_QWORD *)v47;
  }
  if ( v32 != v49 )
    _libc_free((unsigned __int64)v32);
  v34 = v44;
  v35 = &v44[3 * (unsigned int)v45];
  if ( v44 != v35 )
  {
    do
    {
      v36 = *(v35 - 1);
      v35 -= 3;
      if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
        sub_BD60C0(v35);
    }
    while ( v34 != v35 );
    v35 = v44;
  }
  if ( v35 != (__int64 *)v46 )
    _libc_free((unsigned __int64)v35);
  return v41;
}
