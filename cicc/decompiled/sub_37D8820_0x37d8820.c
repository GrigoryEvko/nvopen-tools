// Function: sub_37D8820
// Address: 0x37d8820
//
void __fastcall sub_37D8820(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rbx
  _QWORD *v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // r10d
  __int64 *v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned int v26; // esi
  __int64 *v27; // rax
  __int64 v28; // r8
  char v29; // di
  char *v30; // rax
  char *v31; // rsi
  __int64 v32; // rcx
  char *v33; // rdx
  unsigned int v34; // eax
  _DWORD *v35; // rax
  char *v36; // rax
  int v37; // eax
  int v38; // ecx
  int v39; // r15d
  int v40; // r10d
  __int64 v41; // rax
  _QWORD *v43; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v44; // [rsp+28h] [rbp-D8h]
  _QWORD v45[8]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v46; // [rsp+70h] [rbp-90h] BYREF
  void *s; // [rsp+78h] [rbp-88h]
  _BYTE v48[12]; // [rsp+80h] [rbp-80h]
  char v49; // [rsp+8Ch] [rbp-74h]
  char v50; // [rsp+90h] [rbp-70h] BYREF

  v7 = v45;
  v10 = *(_QWORD *)(a1 + 352);
  s = &v50;
  v11 = *(unsigned int *)(v10 + 40);
  *(_QWORD *)v48 = 8;
  v46 = 0;
  *(_DWORD *)&v48[8] = 0;
  v45[1] = v11 - 1;
  v44 = 0x400000001LL;
  v12 = 1;
  v49 = 1;
  v43 = v45;
  v45[0] = v10;
  do
  {
    while ( 1 )
    {
      v18 = &v7[2 * v12 - 2];
      v19 = v18[1];
      v20 = *v18;
      v18[1] = v19 - 1;
      if ( v19 >= 0 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)(v20 + 32) + 8 * v19);
        v14 = (unsigned int)v44;
        v15 = (unsigned int)v44 + 1LL;
        v16 = *(unsigned int *)(v13 + 40) - 1LL;
        if ( v15 > HIDWORD(v44) )
        {
          sub_C8D5F0((__int64)&v43, v45, v15, 0x10u, a5, a6);
          v7 = v43;
          v14 = (unsigned int)v44;
        }
        v17 = &v7[2 * v14];
        *v17 = v13;
        v17[1] = v16;
        v7 = v43;
        v12 = (unsigned int)(v44 + 1);
        LODWORD(v44) = v44 + 1;
        goto LABEL_5;
      }
      v21 = *(unsigned int *)(a3 + 24);
      a5 = *(_QWORD *)(a3 + 8);
      v12 = (unsigned int)(v44 - 1);
      LODWORD(v44) = v44 - 1;
      if ( (_DWORD)v21 )
      {
        v22 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v23 = (__int64 *)(a5 + 16LL * v22);
        a6 = *v23;
        if ( v20 != *v23 )
        {
          v38 = 1;
          while ( a6 != -4096 )
          {
            v39 = v38 + 1;
            v22 = (v21 - 1) & (v38 + v22);
            v23 = (__int64 *)(a5 + 16LL * v22);
            a6 = *v23;
            if ( v20 == *v23 )
              goto LABEL_9;
            v38 = v39;
          }
          goto LABEL_5;
        }
LABEL_9:
        if ( v23 != (__int64 *)(a5 + 16 * v21) )
          break;
      }
LABEL_5:
      if ( !(_DWORD)v12 )
        goto LABEL_22;
    }
    v24 = *(unsigned int *)(a4 + 24);
    v25 = *(_QWORD *)(a4 + 8);
    if ( (_DWORD)v24 )
    {
      v26 = (v24 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v27 = (__int64 *)(v25 + 72LL * v26);
      v28 = *v27;
      if ( v20 == *v27 )
        goto LABEL_12;
      v37 = 1;
      while ( v28 != -4096 )
      {
        v40 = v37 + 1;
        v41 = ((_DWORD)v24 - 1) & (v26 + v37);
        v26 = v41;
        v27 = (__int64 *)(v25 + 72 * v41);
        v28 = *v27;
        if ( v20 == *v27 )
          goto LABEL_12;
        v37 = v40;
      }
    }
    v27 = (__int64 *)(v25 + 72 * v24);
LABEL_12:
    sub_37D7E90(a1, v23[1], (__int64)&v46, (__int64)(v27 + 1));
    v29 = v49;
    v30 = (char *)s;
    if ( !v49 )
    {
      v31 = (char *)s + 8 * *(unsigned int *)v48;
      if ( s != v31 )
        goto LABEL_14;
      ++v46;
LABEL_17:
      v34 = 4 * (*(_DWORD *)&v48[4] - *(_DWORD *)&v48[8]);
      if ( v34 < 0x20 )
        v34 = 32;
      if ( v34 >= *(_DWORD *)v48 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v48);
        goto LABEL_21;
      }
      sub_C8C990((__int64)&v46, (__int64)v31);
      v12 = (unsigned int)v44;
      v7 = v43;
      goto LABEL_5;
    }
    v31 = (char *)s + 8 * *(unsigned int *)&v48[4];
    if ( s == v31 )
    {
      ++v46;
    }
    else
    {
LABEL_14:
      while ( 1 )
      {
        v32 = *(_QWORD *)v30;
        v33 = v30;
        if ( *(_QWORD *)v30 < 0xFFFFFFFFFFFFFFFELL )
          break;
        v30 += 8;
        if ( v31 == v30 )
          goto LABEL_16;
      }
      if ( v31 != v30 )
      {
        do
        {
          v35 = (_DWORD *)(*a2 + 4LL * *(unsigned int *)(v32 + 24));
          if ( !*v35 )
            *v35 = *(_DWORD *)(v20 + 180);
          v36 = v33 + 8;
          if ( v33 + 8 == v31 )
            break;
          while ( 1 )
          {
            v32 = *(_QWORD *)v36;
            v33 = v36;
            if ( *(_QWORD *)v36 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v36 += 8;
            if ( v31 == v36 )
              goto LABEL_34;
          }
        }
        while ( v36 != v31 );
LABEL_34:
        v29 = v49;
      }
LABEL_16:
      ++v46;
      if ( !v29 )
        goto LABEL_17;
    }
LABEL_21:
    v12 = (unsigned int)v44;
    *(_QWORD *)&v48[4] = 0;
    v7 = v43;
  }
  while ( (_DWORD)v44 );
LABEL_22:
  if ( v7 != v45 )
    _libc_free((unsigned __int64)v7);
  if ( !v49 )
    _libc_free((unsigned __int64)s);
}
