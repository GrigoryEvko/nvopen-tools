// Function: sub_18C4B90
// Address: 0x18c4b90
//
__int64 __fastcall sub_18C4B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 j; // r13
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rdi
  const char *v18; // rax
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // r8
  size_t v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rax
  unsigned int v32; // r12d
  __int64 *v33; // rbx
  __int64 *v34; // r13
  __int64 v35; // r12
  char v36; // al
  __int64 k; // rax
  bool v38; // zf
  __int64 v40; // r13
  __int64 i; // [rsp+10h] [rbp-120h]
  __int64 v43; // [rsp+20h] [rbp-110h]
  __int64 v44; // [rsp+38h] [rbp-F8h]
  __int64 v45; // [rsp+38h] [rbp-F8h]
  _QWORD *v46; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-E8h]
  _QWORD v48[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v49; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v50; // [rsp+68h] [rbp-C8h]
  _BYTE v51[32]; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v52[2]; // [rsp+90h] [rbp-A0h] BYREF
  _BYTE v53[32]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v54; // [rsp+C0h] [rbp-70h]
  __int64 v55; // [rsp+D8h] [rbp-58h]
  __int64 v56; // [rsp+E8h] [rbp-48h]

  v49 = (__int64 *)v51;
  v50 = 0x400000000LL;
  v43 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; i != v43; v43 = *(_QWORD *)(v43 + 8) )
  {
    if ( !v43 )
      BUG();
    for ( j = *(_QWORD *)(v43 + 24); v43 + 16 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      v7 = *(_QWORD *)(j + 24);
      v8 = j + 16;
      if ( v7 != j + 16 )
      {
        v44 = j;
        while ( 1 )
        {
LABEL_8:
          if ( !v7 )
            BUG();
          if ( *(_BYTE *)(v7 - 8) == 29 )
          {
            v46 = *(_QWORD **)(v7 + 16);
            v9 = *(_QWORD *)(v7 - 48);
            v10 = *(_QWORD *)(v9 + 8);
            if ( v10 )
              break;
          }
LABEL_7:
          v7 = *(_QWORD *)(v7 + 8);
          if ( v8 == v7 )
            goto LABEL_14;
        }
        do
        {
          v11 = sub_1648700(v10);
          if ( (unsigned __int8)(*((_BYTE *)v11 + 16) - 25) <= 9u )
          {
LABEL_62:
            v40 = v11[5];
            if ( sub_157F790(v40) && (_QWORD *)v40 != v46 && *(_BYTE *)(sub_157EBA0((__int64)v46) + 16) == 29 )
            {
              v52[0] = (unsigned __int64)v53;
              v52[1] = 0x200000000LL;
              sub_1AAA850(
                v9,
                (unsigned int)&v46,
                1,
                (unsigned int)".1",
                (unsigned int)&unk_4459056,
                (unsigned int)v52,
                0,
                0,
                0);
              if ( (_BYTE *)v52[0] != v53 )
                _libc_free(v52[0]);
            }
            else
            {
              while ( 1 )
              {
                v10 = *(_QWORD *)(v10 + 8);
                if ( !v10 )
                  break;
                v11 = sub_1648700(v10);
                if ( (unsigned __int8)(*((_BYTE *)v11 + 16) - 25) <= 9u )
                  goto LABEL_62;
              }
            }
            goto LABEL_7;
          }
          v10 = *(_QWORD *)(v10 + 8);
        }
        while ( v10 );
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 != v7 )
          goto LABEL_8;
LABEL_14:
        j = v44;
      }
    }
    v12 = (unsigned int)v50;
    if ( (unsigned int)v50 >= HIDWORD(v50) )
    {
      sub_16CD150((__int64)&v49, v51, 0, 8, a5, a6);
      v12 = (unsigned int)v50;
    }
    v49[v12] = v43 - 56;
    LODWORD(v50) = v50 + 1;
  }
  v13 = *(_QWORD *)(a1 + 312);
  if ( v13 == v13 + ((unsigned __int64)*(unsigned int *)(a1 + 320) << 6) )
  {
    v25 = *(unsigned int *)(a1 + 168);
  }
  else
  {
    v45 = v13 + ((unsigned __int64)*(unsigned int *)(a1 + 320) << 6);
    do
    {
      v14 = sub_16321A0(a2, *(_QWORD *)v13, *(_QWORD *)(v13 + 8));
      if ( !v14 )
        sub_16BD130("Invalid function name specified in the input file", 1u);
      v15 = *(_QWORD *)(v14 + 80);
      v16 = v14 + 72;
      if ( v14 + 72 == v15 )
        goto LABEL_71;
      while ( 1 )
      {
        v17 = v15 - 24;
        if ( !v15 )
          v17 = 0;
        v18 = sub_1649960(v17);
        v21 = v20;
        v22 = *(_QWORD *)(v13 + 40);
        if ( v22 == v21 && (!v22 || !memcmp(v18, *(const void **)(v13 + 32), v22)) )
          break;
        v15 = *(_QWORD *)(v15 + 8);
        if ( v16 == v15 )
          goto LABEL_71;
      }
      if ( v16 == v15 )
LABEL_71:
        sub_16BD130("Invalid block name specified in the input file", 1u);
      v23 = v15 - 24;
      v24 = *(unsigned int *)(a1 + 168);
      if ( !v15 )
        v23 = 0;
      if ( (unsigned int)v24 >= *(_DWORD *)(a1 + 172) )
      {
        sub_16CD150(a1 + 160, (const void *)(a1 + 176), 0, 8, v21, v19);
        v24 = *(unsigned int *)(a1 + 168);
      }
      v13 += 64;
      *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v24) = v23;
      v25 = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
      *(_DWORD *)(a1 + 168) = v25;
    }
    while ( v45 != v13 );
  }
  v26 = *(__int64 **)(a1 + 160);
  v27 = &v26[v25];
  if ( v26 == v27 )
  {
    v32 = 0;
  }
  else
  {
    do
    {
      v28 = *v26;
      if ( a2 != *(_QWORD *)(*(_QWORD *)(*v26 + 56) + 40LL) )
        sub_16BD130("Invalid basic block", 1u);
      v48[0] = *v26;
      v46 = v48;
      v47 = 0x200000001LL;
      v29 = sub_157EBA0(v28);
      v30 = 1;
      if ( *(_BYTE *)(v29 + 16) == 29 )
      {
        v31 = *(_QWORD *)(v29 - 24);
        v30 = 2;
        LODWORD(v47) = 2;
        v48[1] = v31;
      }
      sub_1AC09B0((unsigned int)v52, (unsigned int)v48, v30, 0, 0, 0, 0, 0, 0);
      sub_1AC1F00(v52);
      if ( v55 )
        j_j___libc_free_0(v55, v56 - v55);
      j___libc_free_0(v54);
      if ( v46 != v48 )
        _libc_free((unsigned __int64)v46);
      ++v26;
    }
    while ( v27 != v26 );
    v32 = 1;
  }
  if ( *(_BYTE *)(a1 + 304) || LOBYTE(qword_4FADCA0[20]) )
  {
    v33 = v49;
    v34 = &v49[(unsigned int)v50];
    if ( v34 != v49 )
    {
      do
      {
        v35 = *v33;
        sub_15E0C30(*v33);
        v36 = *(_BYTE *)(v35 + 32);
        *(_BYTE *)(v35 + 32) = v36 & 0xF0;
        if ( (v36 & 0x30) != 0 )
          *(_BYTE *)(v35 + 33) |= 0x40u;
        ++v33;
      }
      while ( v34 != v33 );
    }
    for ( k = *(_QWORD *)(a2 + 32); i != k; k = *(_QWORD *)(k + 8) )
    {
      if ( !k )
      {
        MEMORY[0x20] &= 0xFFFFFFF0;
        BUG();
      }
      v38 = (*(_BYTE *)(k - 24) & 0x30) == 0;
      *(_BYTE *)(k - 24) &= 0xF0u;
      if ( !v38 )
        *(_BYTE *)(k - 23) |= 0x40u;
    }
    v32 = 1;
  }
  if ( v49 != (__int64 *)v51 )
    _libc_free((unsigned __int64)v49);
  return v32;
}
