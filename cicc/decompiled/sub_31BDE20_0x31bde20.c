// Function: sub_31BDE20
// Address: 0x31bde20
//
_QWORD *__fastcall sub_31BDE20(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  int v6; // r13d
  __int64 *v7; // r15
  unsigned __int64 v8; // r12
  __int64 *v9; // rax
  __int64 *v10; // r14
  int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 *v15; // r11
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdi
  char v22; // r15
  bool v23; // al
  __int64 v24; // rbx
  __int64 v25; // rax
  int v26; // r10d
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // r10d
  __int64 *v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r13
  __int64 v34; // rdx
  int v35; // eax
  int i; // ebx
  char *v38; // rax
  int v39; // edx
  __int64 v40; // rax
  char *v41; // rax
  int v42; // r11d
  int v43; // edx
  __int64 v44; // rbx
  int v45; // eax
  int v46; // r8d
  int v47; // r8d
  int v48; // [rsp+4h] [rbp-BCh]
  __int64 v49; // [rsp+8h] [rbp-B8h]
  __int64 *v52; // [rsp+38h] [rbp-88h]
  char *v53; // [rsp+40h] [rbp-80h] BYREF
  __int64 v54; // [rsp+48h] [rbp-78h]
  _BYTE v55[112]; // [rsp+50h] [rbp-70h] BYREF

  v7 = (__int64 *)a3;
  v8 = a4;
  v53 = v55;
  v54 = 0x400000000LL;
  if ( a4 > 4 )
    sub_C8D5F0((__int64)&v53, v55, a4, 0x10u, a5, a6);
  v9 = &v7[v8];
  v10 = v7;
  v11 = v6;
  v52 = v9;
  while ( v52 != v10 )
  {
    while ( 1 )
    {
      v24 = *v10;
      v27 = *(_QWORD *)(a2 + 352);
      v28 = *(unsigned int *)(v27 + 24);
      v29 = *(_QWORD *)(v27 + 8);
      if ( !(_DWORD)v28 )
        goto LABEL_25;
      v30 = (v28 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v31 = (__int64 *)(v29 + 16LL * v30);
      v32 = *v31;
      if ( v24 == *v31 )
        break;
      v43 = 1;
      while ( v32 != -4096 )
      {
        a5 = (unsigned int)(v43 + 1);
        v30 = (v28 - 1) & (v43 + v30);
        v31 = (__int64 *)(v29 + 16LL * v30);
        v32 = *v31;
        if ( v24 == *v31 )
          goto LABEL_23;
        v43 = a5;
      }
LABEL_25:
      v34 = (unsigned int)v54;
      a4 = HIDWORD(v54);
      v35 = v54;
      if ( (unsigned int)v54 >= (unsigned __int64)HIDWORD(v54) )
      {
        v44 = v24 | 4;
        if ( HIDWORD(v54) < (unsigned __int64)(unsigned int)v54 + 1 )
        {
          sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 0x10u, a5, a6);
          v34 = (unsigned int)v54;
        }
        a3 = (unsigned __int64)&v53[16 * v34];
        *(_QWORD *)a3 = v44;
        *(_QWORD *)(a3 + 8) = 0;
        LODWORD(v54) = v54 + 1;
        goto LABEL_20;
      }
      a3 = (unsigned __int64)&v53[16 * (unsigned int)v54];
      if ( a3 )
      {
        *(_DWORD *)(a3 + 8) = 0;
        *(_QWORD *)a3 = v24 | 4;
        v35 = v54;
      }
      ++v10;
      LODWORD(v54) = v35 + 1;
      if ( v52 == v10 )
        goto LABEL_29;
    }
LABEL_23:
    if ( v31 == (__int64 *)(v29 + 16 * v28) )
      goto LABEL_25;
    v33 = v31[1];
    if ( !v33 )
      goto LABEL_25;
    v12 = *(unsigned int *)(v27 + 56);
    v13 = *(_QWORD *)(v27 + 40);
    if ( (_DWORD)v12 )
    {
      v14 = (v12 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v15 = (__int64 *)(v13 + 40LL * v14);
      v16 = *v15;
      if ( v33 == *v15 )
      {
LABEL_7:
        if ( v15 != (__int64 *)(v13 + 40 * v12) )
        {
          v17 = *((unsigned int *)v15 + 8);
          v18 = v15[2];
          if ( (_DWORD)v17 )
          {
            v19 = (v17 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v20 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v24 == *v20 )
            {
LABEL_10:
              if ( v20 != (__int64 *)(v18 + 16 * v17) )
              {
                v11 = *((_DWORD *)v20 + 2);
                v22 = 1;
                goto LABEL_12;
              }
            }
            else
            {
              v45 = 1;
              while ( v21 != -4096 )
              {
                v46 = v45 + 1;
                v19 = (v17 - 1) & (v45 + v19);
                v20 = (__int64 *)(v18 + 16LL * v19);
                v21 = *v20;
                if ( v24 == *v20 )
                  goto LABEL_10;
                v45 = v46;
              }
            }
          }
        }
      }
      else
      {
        v42 = 1;
        while ( v16 != -4096 )
        {
          v47 = v42 + 1;
          v14 = (v12 - 1) & (v42 + v14);
          v15 = (__int64 *)(v13 + 40LL * v14);
          v16 = *v15;
          if ( v33 == *v15 )
            goto LABEL_7;
          v42 = v47;
        }
      }
    }
    v22 = 0;
LABEL_12:
    v23 = sub_318B630(*v10);
    if ( v24 && v23 && (*(_DWORD *)(v24 + 8) != 37 || sub_318B6C0(v24)) )
    {
      if ( sub_318B670(v24) )
      {
        v24 = sub_318B680(v24);
      }
      else if ( *(_DWORD *)(v24 + 8) == 37 )
      {
        v24 = sub_318B6C0(v24);
      }
    }
    v25 = *sub_318EB80(v24);
    if ( *(_BYTE *)(v25 + 8) != 17 )
    {
      v26 = 1;
LABEL_35:
      for ( i = 0; i != v26; ++i )
      {
        a4 = 0xFFFFFFFFLL;
        if ( v22 )
          a4 = (unsigned int)(v11 + i);
        v40 = (unsigned int)v54;
        v39 = v54;
        if ( (unsigned int)v54 < (unsigned __int64)HIDWORD(v54) )
        {
          v38 = &v53[16 * (unsigned int)v54];
          if ( v38 )
          {
            *((_DWORD *)v38 + 2) = a4;
            *(_QWORD *)v38 = v33 & 0xFFFFFFFFFFFFFFFBLL;
            v39 = v54;
          }
          a3 = (unsigned int)(v39 + 1);
          LODWORD(v54) = a3;
        }
        else
        {
          a3 = (unsigned int)v54 + 1LL;
          a6 = (unsigned int)a4;
          a5 = v33 & 0xFFFFFFFFFFFFFFFBLL;
          if ( HIDWORD(v54) < a3 )
          {
            v48 = v26;
            v49 = (unsigned int)a4;
            sub_C8D5F0((__int64)&v53, v55, a3, 0x10u, a5, (unsigned int)a4);
            v40 = (unsigned int)v54;
            v26 = v48;
            a6 = v49;
            a5 = v33 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v41 = &v53[16 * v40];
          *(_QWORD *)v41 = a5;
          *((_QWORD *)v41 + 1) = a6;
          LODWORD(v54) = v54 + 1;
        }
      }
      goto LABEL_20;
    }
    v26 = *(_DWORD *)(v25 + 32);
    if ( v26 )
      goto LABEL_35;
LABEL_20:
    ++v10;
  }
LABEL_29:
  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  if ( (_DWORD)v54 )
    sub_31BC1F0((__int64)a1, &v53, a3, a4, a5, a6);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return a1;
}
