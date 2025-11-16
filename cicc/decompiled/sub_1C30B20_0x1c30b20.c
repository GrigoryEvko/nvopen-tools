// Function: sub_1C30B20
// Address: 0x1c30b20
//
__int64 __fastcall sub_1C30B20(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  char *v7; // rsi
  __int64 v8; // rdi
  size_t v9; // r15
  void *v10; // r9
  __int64 v11; // rax
  int v12; // r9d
  unsigned int v13; // ebx
  _BYTE *v14; // rax
  __int64 v15; // rdx
  const void *v16; // r8
  size_t v17; // rbx
  _BYTE *v18; // rsi
  int v19; // ebx
  __int64 p_src; // rdi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  void *v24; // [rsp+8h] [rbp-118h]
  const void *v25; // [rsp+8h] [rbp-118h]
  _QWORD v26[2]; // [rsp+10h] [rbp-110h] BYREF
  _QWORD v27[2]; // [rsp+20h] [rbp-100h] BYREF
  void *src; // [rsp+30h] [rbp-F0h] BYREF
  size_t n; // [rsp+38h] [rbp-E8h]
  _BYTE *v30; // [rsp+40h] [rbp-E0h] BYREF
  _BYTE *v31; // [rsp+48h] [rbp-D8h]
  int v32; // [rsp+50h] [rbp-D0h]
  const void **v33; // [rsp+58h] [rbp-C8h]
  _BYTE *v34; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+68h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+70h] [rbp-B0h] BYREF

  v34 = v36;
  v35 = 0x8000000000LL;
  v4 = sub_1626D20(a2);
  if ( !v4 )
    goto LABEL_30;
  if ( *(_BYTE *)v4 == 15 || (v4 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8))) != 0 )
  {
    v5 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
    if ( v5 )
    {
      v7 = (char *)sub_161E970(v5);
      if ( v7 )
        goto LABEL_6;
    }
LABEL_30:
    LOBYTE(v30) = 0;
    LODWORD(v9) = 0;
    LODWORD(v8) = v35;
    src = &v30;
    n = 0;
    goto LABEL_10;
  }
  v6 = 0;
  v7 = (char *)byte_3F871B3;
LABEL_6:
  src = &v30;
  sub_1C30A70((__int64 *)&src, v7, (__int64)&v7[v6]);
  v8 = (unsigned int)v35;
  v9 = n;
  v10 = src;
  if ( n > HIDWORD(v35) - (unsigned __int64)(unsigned int)v35 )
  {
    v24 = src;
    sub_16CD150((__int64)&v34, v36, n + (unsigned int)v35, 1, (int)&v30, (int)src);
    v8 = (unsigned int)v35;
    v10 = v24;
  }
  if ( v9 )
  {
    memcpy(&v34[v8], v10, v9);
    LODWORD(v8) = v35;
  }
LABEL_10:
  LODWORD(v35) = v9 + v8;
  if ( src != &v30 )
    j_j___libc_free_0(src, v30 + 1);
  v26[0] = v27;
  v26[1] = 0;
  LOBYTE(v27[0]) = 0;
  v32 = 1;
  v31 = 0;
  v30 = 0;
  n = 0;
  src = &unk_49EFBE0;
  v33 = (const void **)v26;
  v11 = sub_1626D20(a2);
  if ( v11 && (v13 = *(_DWORD *)(v11 + 24)) != 0 )
  {
    if ( v30 == v31 )
    {
      p_src = sub_16E7EE0((__int64)&src, "(", 1u);
    }
    else
    {
      *v31 = 40;
      p_src = (__int64)&src;
      ++v31;
    }
    v22 = sub_16E7A90(p_src, v13);
    v23 = *(_BYTE **)(v22 + 24);
    if ( *(_BYTE **)(v22 + 16) == v23 )
    {
      sub_16E7EE0(v22, ")", 1u);
    }
    else
    {
      *v23 = 41;
      ++*(_QWORD *)(v22 + 24);
    }
  }
  else
  {
    if ( (unsigned __int64)(v30 - v31) > 1 )
    {
      *(_WORD *)v31 = 10536;
      v14 = v31 + 2;
      v31 += 2;
      goto LABEL_16;
    }
    sub_16E7EE0((__int64)&src, "()", 2u);
  }
  v14 = v31;
LABEL_16:
  if ( (_BYTE *)n != v14 )
    sub_16E7BA0((__int64 *)&src);
  v15 = (unsigned int)v35;
  v16 = *v33;
  v17 = (size_t)v33[1];
  if ( v17 > HIDWORD(v35) - (unsigned __int64)(unsigned int)v35 )
  {
    v25 = *v33;
    sub_16CD150((__int64)&v34, v36, v17 + (unsigned int)v35, 1, (int)v16, v12);
    v15 = (unsigned int)v35;
    v16 = v25;
  }
  v18 = v34;
  if ( v17 )
  {
    memcpy(&v34[v15], v16, v17);
    v18 = v34;
    LODWORD(v15) = v35;
  }
  v19 = v15 + v17;
  LODWORD(v35) = v19;
  *(_QWORD *)a1 = a1 + 16;
  if ( v18 )
  {
    sub_1C30A70((__int64 *)a1, v18, (__int64)&v18[v19]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  sub_16E7BC0((__int64 *)&src);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0], v27[0] + 1LL);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  return a1;
}
