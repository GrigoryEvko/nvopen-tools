// Function: sub_2A3BF30
// Address: 0x2a3bf30
//
__int64 __fastcall sub_2A3BF30(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rcx
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // r9
  int v8; // r14d
  __int64 v9; // rbx
  unsigned int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // r10
  __int64 v13; // rdx
  int v15; // eax
  int v16; // edi
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  int v23; // edx
  __int64 v24; // rdi
  __int64 *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rdi
  char *v30; // rdi
  int v31; // edx
  int v32; // esi
  unsigned int v33; // edx
  int v34; // r11d
  __int64 v35; // r10
  unsigned __int64 v36; // rdx
  __int64 v37; // rdi
  int v38; // edx
  int v39; // edx
  unsigned int v40; // r13d
  int v41; // r10d
  __int64 v42; // rsi
  __int64 v43; // [rsp+8h] [rbp-168h]
  __int64 *v44; // [rsp+18h] [rbp-158h]
  __int64 *v45; // [rsp+18h] [rbp-158h]
  __int64 v46; // [rsp+20h] [rbp-150h] BYREF
  _BYTE *v47; // [rsp+28h] [rbp-148h]
  __int64 v48; // [rsp+30h] [rbp-140h]
  _BYTE v49[16]; // [rsp+38h] [rbp-138h] BYREF
  _BYTE *v50; // [rsp+48h] [rbp-128h]
  __int64 v51; // [rsp+50h] [rbp-120h]
  _BYTE v52[16]; // [rsp+58h] [rbp-118h] BYREF
  _BYTE *v53; // [rsp+68h] [rbp-108h]
  __int64 v54; // [rsp+70h] [rbp-100h]
  _BYTE v55[16]; // [rsp+78h] [rbp-F8h] BYREF
  _BYTE *v56; // [rsp+88h] [rbp-E8h]
  __int64 v57; // [rsp+90h] [rbp-E0h]
  _BYTE v58[24]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v59; // [rsp+B0h] [rbp-C0h] BYREF
  char v60[8]; // [rsp+B8h] [rbp-B8h] BYREF
  char *v61; // [rsp+C0h] [rbp-B0h]
  char v62; // [rsp+D0h] [rbp-A0h] BYREF
  char *v63; // [rsp+E0h] [rbp-90h]
  char v64; // [rsp+F0h] [rbp-80h] BYREF
  char *v65; // [rsp+100h] [rbp-70h]
  char v66; // [rsp+110h] [rbp-60h] BYREF
  char *v67; // [rsp+120h] [rbp-50h]
  char v68; // [rsp+130h] [rbp-40h] BYREF

  v2 = a2;
  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_39;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v5 - 1;
  v8 = 1;
  v9 = 0;
  v10 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = v6 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( v4 == *(_QWORD *)v11 )
  {
LABEL_3:
    v13 = *(unsigned int *)(v11 + 8);
    return 144 * v13 + *(_QWORD *)(a1 + 32) + 8;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = v7 & (v8 + v10);
    v11 = v6 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v4 == *(_QWORD *)v11 )
      goto LABEL_3;
    ++v8;
  }
  v15 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v11;
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_39:
    v44 = v2;
    sub_2A3BD50(a1, 2 * v5);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      v2 = v44;
      v33 = (v31 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v7 + 16LL * v33;
      v17 = *(_QWORD *)v9;
      if ( v4 != *(_QWORD *)v9 )
      {
        v34 = 1;
        v35 = 0;
        while ( v17 != -4096 )
        {
          if ( !v35 && v17 == -8192 )
            v35 = v9;
          v33 = v32 & (v34 + v33);
          v9 = v7 + 16LL * v33;
          v17 = *(_QWORD *)v9;
          if ( v4 == *(_QWORD *)v9 )
            goto LABEL_15;
          ++v34;
        }
        if ( v35 )
          v9 = v35;
      }
      goto LABEL_15;
    }
    goto LABEL_66;
  }
  v17 = v5 >> 3;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= (unsigned int)v17 )
  {
    v45 = v2;
    sub_2A3BD50(a1, v5);
    v38 = *(_DWORD *)(a1 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v7 = 0;
      v40 = v39 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v2 = v45;
      v41 = 1;
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v17 + 16LL * v40;
      v42 = *(_QWORD *)v9;
      if ( v4 != *(_QWORD *)v9 )
      {
        while ( v42 != -4096 )
        {
          if ( !v7 && v42 == -8192 )
            v7 = v9;
          v40 = v39 & (v41 + v40);
          v9 = v17 + 16LL * v40;
          v42 = *(_QWORD *)v9;
          if ( v4 == *(_QWORD *)v9 )
            goto LABEL_15;
          ++v41;
        }
        if ( v7 )
          v9 = v7;
      }
      goto LABEL_15;
    }
LABEL_66:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  *(_DWORD *)(v9 + 8) = 0;
  v48 = 0x200000000LL;
  v51 = 0x200000000LL;
  v54 = 0x200000000LL;
  v57 = 0x200000000LL;
  v18 = *v2;
  v56 = v58;
  v59 = v18;
  v46 = 0;
  v47 = v49;
  v50 = v52;
  v53 = v55;
  sub_2A3B9E0((__int64)v60, (__int64)&v46, v18, (__int64)v2, v17, v7);
  v21 = *(unsigned int *)(a1 + 40);
  v22 = v21 + 1;
  v23 = *(_DWORD *)(a1 + 40);
  if ( v21 + 1 > *(unsigned int *)(a1 + 44) )
  {
    v36 = *(_QWORD *)(a1 + 32);
    v37 = a1 + 32;
    if ( v36 > (unsigned __int64)&v59
      || (v43 = *(_QWORD *)(a1 + 32), v21 = v36 + 144 * v21, (unsigned __int64)&v59 >= v21) )
    {
      sub_2A3BC20(v37, v22, v36, v21, v19, v20);
      v21 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v25 = &v59;
      v23 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2A3BC20(v37, v22, v36, v21, v19, v20);
      v24 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v25 = (__int64 *)&v60[v24 - 8 - v43];
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = &v59;
  }
  v26 = 144 * v21;
  v27 = v26 + v24;
  if ( v27 )
  {
    v28 = *v25;
    v29 = v27 + 8;
    *(_QWORD *)(v29 - 8) = *v25;
    sub_2A3B9E0(v29, (__int64)(v25 + 1), v28, v26, v19, v20);
    v23 = *(_DWORD *)(a1 + 40);
  }
  v30 = v67;
  *(_DWORD *)(a1 + 40) = v23 + 1;
  if ( v30 != &v68 )
    _libc_free((unsigned __int64)v30);
  if ( v65 != &v66 )
    _libc_free((unsigned __int64)v65);
  if ( v63 != &v64 )
    _libc_free((unsigned __int64)v63);
  if ( v61 != &v62 )
    _libc_free((unsigned __int64)v61);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v9 + 8) = v13;
  return 144 * v13 + *(_QWORD *)(a1 + 32) + 8;
}
