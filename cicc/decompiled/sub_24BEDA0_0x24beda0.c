// Function: sub_24BEDA0
// Address: 0x24beda0
//
__int64 __fastcall sub_24BEDA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // r10d
  unsigned int i; // eax
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rsi
  int v23; // eax
  unsigned __int64 v24; // rdi
  char *v25; // r13
  __int64 *v26; // rbx
  char v27; // bl
  void *v28; // rsi
  __int64 v29; // r13
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rdi
  char *v34; // rsi
  __int64 v35; // [rsp+8h] [rbp-A8h]
  __int64 v36; // [rsp+18h] [rbp-98h] BYREF
  __int64 v37; // [rsp+20h] [rbp-90h] BYREF
  void **v38; // [rsp+28h] [rbp-88h]
  __int64 v39; // [rsp+30h] [rbp-80h]
  __int64 v40; // [rsp+38h] [rbp-78h]
  void *v41; // [rsp+40h] [rbp-70h] BYREF
  char v42; // [rsp+48h] [rbp-68h]
  char *v43; // [rsp+50h] [rbp-60h] BYREF
  char *v44; // [rsp+58h] [rbp-58h]
  __int64 v45; // [rsp+60h] [rbp-50h]
  int v46; // [rsp+68h] [rbp-48h]
  char v47; // [rsp+6Ch] [rbp-44h]
  _BYTE v48[64]; // [rsp+70h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F8D9A8, a3);
  v8 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v35 = v8;
  if ( !(_DWORD)v9 )
    goto LABEL_52;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F81450 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_52;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 )
  {
LABEL_52:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8;
  }
  v16 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  if ( (_BYTE)qword_4FEC9E8 )
    goto LABEL_47;
  if ( (unsigned __int8)sub_B2D610(a3, 47) )
    goto LABEL_47;
  v40 = v15;
  v17 = *(_QWORD *)(a3 + 80);
  v38 = (void **)(v7 + 8);
  v18 = a3 + 72;
  v37 = a3;
  v39 = v35 + 8;
  v41 = (void *)(v16 + 8);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  if ( a3 + 72 == v17 )
  {
LABEL_47:
    v28 = (void *)(a1 + 32);
    v29 = a1 + 80;
LABEL_48:
    *(_QWORD *)(a1 + 8) = v28;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v29;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  do
  {
    v19 = v17;
    v17 = *(_QWORD *)(v17 + 8);
    v20 = *(_QWORD *)(v19 + 32);
    v21 = v19 + 24;
LABEL_15:
    while ( v21 != v20 )
    {
      while ( 1 )
      {
        v22 = v20;
        v20 = *(_QWORD *)(v20 + 8);
        v23 = *(unsigned __int8 *)(v22 - 24);
        if ( v23 == 85 )
        {
          v31 = *(_QWORD *)(v22 - 56);
          if ( v31
            && !*(_BYTE *)v31
            && *(_QWORD *)(v31 + 24) == *(_QWORD *)(v22 + 56)
            && (v32 = *(_DWORD *)(v31 + 36), v32 <= 0xF5)
            && v32 > 0xED )
          {
            switch ( v32 )
            {
              case 0xEEu:
              case 0xF0u:
              case 0xF1u:
              case 0xF3u:
              case 0xF5u:
                v33 = v22 - 24;
                if ( **(_BYTE **)(v22 + 32 * (2LL - (*(_DWORD *)(v22 - 20) & 0x7FFFFFF)) - 24) != 17 )
                {
                  v36 = v22 - 24;
                  v34 = v44;
                  if ( v44 == (char *)v45 )
                  {
                    sub_24BBE90((unsigned __int64 *)&v43, v44, &v36);
                  }
                  else
                  {
                    if ( v44 )
                    {
                      *(_QWORD *)v44 = v33;
                      v34 = v44;
                    }
                    v44 = v34 + 8;
                  }
                }
                break;
              default:
                goto LABEL_36;
            }
          }
          else
          {
LABEL_36:
            sub_24BC370(&v37, v22 - 24);
          }
          goto LABEL_15;
        }
        if ( (unsigned int)(v23 - 29) <= 0x38 )
          break;
        if ( (unsigned int)(v23 - 86) > 0xA )
          goto LABEL_55;
        if ( v21 == v20 )
          goto LABEL_20;
      }
      if ( (unsigned int)(v23 - 30) > 0x36 )
LABEL_55:
        BUG();
    }
LABEL_20:
    ;
  }
  while ( v18 != v17 );
  v24 = (unsigned __int64)v43;
  v25 = v44;
  if ( v43 == v44 )
  {
    v27 = v42;
  }
  else
  {
    v26 = (__int64 *)v43;
    do
    {
      if ( (unsigned __int8)sub_24BCCA0(&v37, *v26) )
        v42 = 1;
      ++v26;
    }
    while ( v25 != (char *)v26 );
    v24 = (unsigned __int64)v43;
    v27 = v42;
  }
  if ( v24 )
    j_j___libc_free_0(v24);
  v28 = (void *)(a1 + 32);
  v29 = a1 + 80;
  if ( !v27 )
    goto LABEL_48;
  v38 = &v41;
  v39 = 0x100000002LL;
  v41 = &unk_4F81450;
  LODWORD(v40) = 0;
  BYTE4(v40) = 1;
  v43 = 0;
  v44 = v48;
  v45 = 2;
  v46 = 0;
  v47 = 1;
  v37 = 1;
  sub_C8CF70(a1, v28, 2, (__int64)&v41, (__int64)&v37);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v48, (__int64)&v43);
  if ( !v47 )
  {
    _libc_free((unsigned __int64)v44);
    if ( BYTE4(v40) )
      return a1;
LABEL_50:
    _libc_free((unsigned __int64)v38);
    return a1;
  }
  if ( !BYTE4(v40) )
    goto LABEL_50;
  return a1;
}
