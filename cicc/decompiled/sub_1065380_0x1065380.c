// Function: sub_1065380
// Address: 0x1065380
//
unsigned __int64 *__fastcall sub_1065380(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r12
  __int64 v11; // rcx
  int v12; // r11d
  __int64 *v13; // r8
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  int v19; // ebx
  __int64 *v20; // rax
  __int64 *j; // rdx
  __int64 v22; // rbx
  __int64 v23; // rsi
  _QWORD *v24; // rax
  bool v26; // zf
  unsigned int v27; // eax
  __int64 v28; // rdx
  int v29; // eax
  int v30; // edx
  int v31; // eax
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34; // rdi
  int v35; // r10d
  int v36; // eax
  int v37; // eax
  __int64 v38; // rsi
  unsigned int v39; // ebx
  __int64 *v40; // rdi
  __int64 v41; // [rsp+0h] [rbp-160h]
  __int64 *v43; // [rsp+18h] [rbp-148h]
  __int64 v44; // [rsp+20h] [rbp-140h]
  __int64 *i; // [rsp+28h] [rbp-138h]
  _QWORD *v46; // [rsp+30h] [rbp-130h]
  unsigned __int64 v47; // [rsp+40h] [rbp-120h] BYREF
  char *v48; // [rsp+48h] [rbp-118h]
  __int64 v49; // [rsp+50h] [rbp-110h]
  int v50; // [rsp+58h] [rbp-108h]
  char v51; // [rsp+5Ch] [rbp-104h]
  char v52; // [rsp+60h] [rbp-100h] BYREF
  __int64 *v53; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-B8h]
  _BYTE v55[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v6 = a2;
  v7 = *(unsigned int *)(a2 + 336);
  v53 = (__int64 *)v55;
  v54 = 0x1000000000LL;
  v8 = *(__int64 **)(a2 + 328);
  v41 = a2 + 8;
  v43 = &v8[v7];
  for ( i = v8; v43 != i; ++i )
  {
    v9 = *(_DWORD *)(v6 + 32);
    v10 = *i;
    if ( v9 )
    {
      v11 = *(_QWORD *)(v6 + 16);
      v12 = 1;
      v13 = 0;
      v14 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v15 = (__int64 *)(v11 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
      {
LABEL_4:
        v44 = v15[1];
        goto LABEL_5;
      }
      while ( v16 != -4096 )
      {
        if ( !v13 && v16 == -8192 )
          v13 = v15;
        a6 = (unsigned int)(v12 + 1);
        v14 = (v9 - 1) & (v12 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v10 == *v15 )
          goto LABEL_4;
        ++v12;
      }
      if ( !v13 )
        v13 = v15;
      v29 = *(_DWORD *)(v6 + 24);
      ++*(_QWORD *)(v6 + 8);
      v30 = v29 + 1;
      if ( 4 * (v29 + 1) < 3 * v9 )
      {
        v11 = v9 >> 3;
        if ( v9 - *(_DWORD *)(v6 + 28) - v30 <= (unsigned int)v11 )
        {
          sub_1062330(v41, v9);
          v36 = *(_DWORD *)(v6 + 32);
          if ( !v36 )
          {
LABEL_72:
            ++*(_DWORD *)(v6 + 24);
            BUG();
          }
          v37 = v36 - 1;
          v38 = *(_QWORD *)(v6 + 16);
          a6 = 1;
          v39 = v37 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v30 = *(_DWORD *)(v6 + 24) + 1;
          v40 = 0;
          v13 = (__int64 *)(v38 + 16LL * v39);
          v11 = *v13;
          if ( v10 != *v13 )
          {
            while ( v11 != -4096 )
            {
              if ( !v40 && v11 == -8192 )
                v40 = v13;
              v39 = v37 & (a6 + v39);
              v13 = (__int64 *)(v38 + 16LL * v39);
              v11 = *v13;
              if ( v10 == *v13 )
                goto LABEL_44;
              a6 = (unsigned int)(a6 + 1);
            }
            if ( v40 )
              v13 = v40;
          }
        }
        goto LABEL_44;
      }
    }
    else
    {
      ++*(_QWORD *)(v6 + 8);
    }
    sub_1062330(v41, 2 * v9);
    v31 = *(_DWORD *)(v6 + 32);
    if ( !v31 )
      goto LABEL_72;
    v11 = (unsigned int)(v31 - 1);
    v32 = *(_QWORD *)(v6 + 16);
    v33 = v11 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v30 = *(_DWORD *)(v6 + 24) + 1;
    v13 = (__int64 *)(v32 + 16LL * v33);
    v34 = *v13;
    if ( v10 != *v13 )
    {
      v35 = 1;
      a6 = 0;
      while ( v34 != -4096 )
      {
        if ( !a6 && v34 == -8192 )
          a6 = (__int64)v13;
        v33 = v11 & (v35 + v33);
        v13 = (__int64 *)(v32 + 16LL * v33);
        v34 = *v13;
        if ( v10 == *v13 )
          goto LABEL_44;
        ++v35;
      }
      if ( a6 )
        v13 = (__int64 *)a6;
    }
LABEL_44:
    *(_DWORD *)(v6 + 24) = v30;
    if ( *v13 != -4096 )
      --*(_DWORD *)(v6 + 28);
    *v13 = v10;
    v13[1] = 0;
    v44 = 0;
LABEL_5:
    v17 = *(unsigned int *)(v10 + 12);
    v18 = (unsigned int)v54;
    v19 = *(_DWORD *)(v10 + 12);
    if ( v17 == (unsigned int)v54 )
    {
      v19 = v54;
    }
    else
    {
      if ( v17 >= (unsigned int)v54 )
      {
        if ( v17 > HIDWORD(v54) )
        {
          sub_C8D5F0((__int64)&v53, v55, *(unsigned int *)(v10 + 12), 8u, (__int64)v13, a6);
          v18 = (unsigned int)v54;
        }
        v20 = &v53[v18];
        for ( j = &v53[v17]; j != v20; ++v20 )
        {
          if ( v20 )
            *v20 = 0;
        }
      }
      LODWORD(v54) = v17;
    }
    if ( v19 )
    {
      v22 = 0;
      do
      {
        v23 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + v22 * 8);
        v48 = &v52;
        v47 = 0;
        v49 = 8;
        v50 = 0;
        v51 = 1;
        v24 = sub_10648E0(v6, v23, (__int64)&v47, v11, (__int64)v13, a6);
        if ( !v51 )
        {
          v46 = v24;
          _libc_free(v48, v23);
          v24 = v46;
        }
        v53[v22++] = (__int64)v24;
      }
      while ( v22 != v17 );
      v17 = (unsigned int)v54;
    }
    a2 = v44;
    sub_BD0A20(&v47, v44, v53, v17, (*(_DWORD *)(v10 + 8) & 0x200) != 0);
    if ( (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = v47 & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_22;
    }
    a2 = v44;
    sub_1063C40(*(_QWORD *)(v6 + 632), v44);
  }
  ++*(_QWORD *)(v6 + 472);
  v26 = *(_BYTE *)(v6 + 500) == 0;
  *(_DWORD *)(v6 + 336) = 0;
  if ( v26 )
  {
    v27 = 4 * (*(_DWORD *)(v6 + 492) - *(_DWORD *)(v6 + 496));
    v28 = *(unsigned int *)(v6 + 488);
    if ( v27 < 0x20 )
      v27 = 32;
    if ( v27 < (unsigned int)v28 )
    {
      sub_C8C990(v6 + 472, a2);
      goto LABEL_32;
    }
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(v6 + 480), -1, 8 * v28);
  }
  *(_QWORD *)(v6 + 492) = 0;
LABEL_32:
  *a1 = 1;
LABEL_22:
  if ( v53 != (__int64 *)v55 )
    _libc_free(v53, a2);
  return a1;
}
