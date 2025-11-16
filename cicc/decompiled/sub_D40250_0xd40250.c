// Function: sub_D40250
// Address: 0xd40250
//
__int64 __fastcall sub_D40250(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  unsigned __int64 *v8; // r12
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __int64 *v16; // rsi
  unsigned __int64 v17; // rax
  char *v18; // r13
  __int64 v19; // rdi
  unsigned __int64 v20; // r10
  int v21; // edx
  __int64 *v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rdi
  char *v25; // rdi
  __int64 v26; // rsi
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  unsigned int v30; // eax
  unsigned __int64 v31; // rsi
  int v32; // r9d
  unsigned __int64 *v33; // r8
  unsigned __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // r15
  char *v37; // r13
  __int64 v38; // rdi
  int v39; // edx
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  int v43; // r8d
  unsigned int v44; // r15d
  unsigned __int64 *v45; // rdi
  unsigned __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // [rsp+0h] [rbp-D0h]
  int v49; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v50; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v51; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+30h] [rbp-A0h]
  __int64 v54; // [rsp+38h] [rbp-98h]
  _BYTE *v55; // [rsp+40h] [rbp-90h]
  __int64 v56; // [rsp+48h] [rbp-88h]
  _BYTE v57[16]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v58; // [rsp+60h] [rbp-70h] BYREF
  char v59[8]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v60; // [rsp+70h] [rbp-60h]
  unsigned int v61; // [rsp+80h] [rbp-50h]
  char *v62; // [rsp+88h] [rbp-48h]
  char v63; // [rsp+98h] [rbp-38h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (v4 ^ (v4 >> 9));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + (v12 << 6) + 8;
  }
  while ( v11 != -4 )
  {
    if ( v11 == -16 && !v8 )
      v8 = (unsigned __int64 *)v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == v4 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = (unsigned __int64 *)v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_27:
    sub_D40070(a1, 2 * v5);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v30 = (v27 - 1) & (v4 ^ (v4 >> 9));
      v8 = (unsigned __int64 *)(v29 + 16LL * v30);
      v31 = *v8;
      if ( v4 != *v8 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -4 )
        {
          if ( !v33 && v31 == -16 )
            v33 = v8;
          v30 = v28 & (v32 + v30);
          v8 = (unsigned __int64 *)(v29 + 16LL * v30);
          v31 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v32;
        }
        if ( v33 )
          v8 = v33;
      }
      goto LABEL_15;
    }
    goto LABEL_60;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_D40070(a1, v5);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      v43 = 1;
      v44 = v41 & (v4 ^ (v4 >> 9));
      v45 = 0;
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (unsigned __int64 *)(v42 + 16LL * v44);
      v46 = *v8;
      if ( *v8 != v4 )
      {
        while ( v46 != -4 )
        {
          if ( v46 == -16 && !v45 )
            v45 = v8;
          v44 = v41 & (v43 + v44);
          v8 = (unsigned __int64 *)(v42 + 16LL * v44);
          v46 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v43;
        }
        if ( v45 )
          v8 = v45;
      }
      goto LABEL_15;
    }
LABEL_60:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v16 = &v51;
  *((_DWORD *)v8 + 2) = 0;
  v56 = 0x100000000LL;
  v17 = *a2;
  v18 = (char *)&v58;
  v51 = 0;
  v58 = v17;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = v57;
  sub_D38730((__int64)v59, (__int64)&v51);
  v19 = *(unsigned int *)(a1 + 40);
  v20 = v19 + 1;
  v21 = *(_DWORD *)(a1 + 40);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v34 = *(_QWORD *)(a1 + 32);
    v35 = a1 + 32;
    v36 = a1 + 48;
    if ( v34 > (unsigned __int64)&v58 || (unsigned __int64)&v58 >= v34 + (v19 << 6) )
    {
      v16 = (__int64 *)sub_C8D7D0(a1 + 32, a1 + 48, v20, 0x40u, &v50, v35);
      sub_D39F20(a1 + 32, (__int64)v16);
      v47 = *(_QWORD *)(a1 + 32);
      v22 = v16;
      if ( v36 == v47 )
      {
        v19 = *(unsigned int *)(a1 + 40);
        *(_DWORD *)(a1 + 44) = v50;
      }
      else
      {
        v49 = v50;
        _libc_free(v47, v16);
        v22 = v16;
        v19 = *(unsigned int *)(a1 + 40);
        *(_DWORD *)(a1 + 44) = v49;
      }
      *(_QWORD *)(a1 + 32) = v16;
      v21 = v19;
    }
    else
    {
      v37 = &v59[-v34 - 8];
      v16 = (__int64 *)sub_C8D7D0(a1 + 32, a1 + 48, v20, 0x40u, &v50, v35);
      sub_D39F20(a1 + 32, (__int64)v16);
      v38 = *(_QWORD *)(a1 + 32);
      v39 = v50;
      v22 = v16;
      if ( v36 == v38 )
      {
        *(_QWORD *)(a1 + 32) = v16;
        *(_DWORD *)(a1 + 44) = v39;
      }
      else
      {
        v48 = v50;
        _libc_free(v38, v16);
        v22 = v16;
        *(_QWORD *)(a1 + 32) = v16;
        *(_DWORD *)(a1 + 44) = v48;
      }
      v19 = *(unsigned int *)(a1 + 40);
      v18 = &v37[(_QWORD)v22];
      v21 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v22 = *(__int64 **)(a1 + 32);
  }
  v23 = &v22[8 * v19];
  if ( v23 )
  {
    v16 = (__int64 *)(v18 + 8);
    v24 = (__int64)(v23 + 1);
    *(_QWORD *)(v24 - 8) = *(_QWORD *)v18;
    sub_D38730(v24, (__int64)(v18 + 8));
    v21 = *(_DWORD *)(a1 + 40);
  }
  v25 = v62;
  *(_DWORD *)(a1 + 40) = v21 + 1;
  if ( v25 != &v63 )
    _libc_free(v25, v16);
  v26 = 8LL * v61;
  sub_C7D6A0(v60, v26, 8);
  if ( v55 != v57 )
    _libc_free(v55, v26);
  sub_C7D6A0(v52, 8LL * (unsigned int)v54, 8);
  v12 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + (v12 << 6) + 8;
}
