// Function: sub_14C4510
// Address: 0x14c4510
//
__int64 __fastcall sub_14C4510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // r15
  __int64 v9; // r11
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v15; // r15
  const void *v16; // rax
  signed __int64 v17; // r15
  const void *v18; // r11
  __int64 v19; // r9
  _BYTE *v20; // r10
  int v21; // edx
  _BYTE *v22; // rdi
  unsigned __int64 v23; // r9
  size_t v24; // r8
  _QWORD *v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // esi
  __int64 *v28; // rdi
  _BYTE *src; // [rsp+0h] [rbp-C0h]
  size_t n; // [rsp+8h] [rbp-B8h]
  size_t na; // [rsp+8h] [rbp-B8h]
  int v32; // [rsp+10h] [rbp-B0h]
  const void *v33; // [rsp+10h] [rbp-B0h]
  int v34; // [rsp+18h] [rbp-A8h]
  int v35; // [rsp+18h] [rbp-A8h]
  __int64 v36; // [rsp+20h] [rbp-A0h]
  __int64 v37; // [rsp+20h] [rbp-A0h]
  void *dest; // [rsp+28h] [rbp-98h]
  _BYTE *v39; // [rsp+30h] [rbp-90h] BYREF
  __int64 v40; // [rsp+38h] [rbp-88h]
  _BYTE v41[32]; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v42; // [rsp+60h] [rbp-60h] BYREF
  __int64 v43; // [rsp+68h] [rbp-58h]
  _BYTE v44[80]; // [rsp+70h] [rbp-50h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return 0;
  v5 = sub_14C43E0(a1, a2, a3);
  v6 = sub_146F1B0(a2, v5);
  v7 = *(_WORD *)(v6 + 24);
  if ( a1 != v5 && (unsigned __int16)(v7 - 1) <= 2u )
  {
    do
    {
      v6 = *(_QWORD *)(v6 + 32);
      v7 = *(_WORD *)(v6 + 24);
    }
    while ( (unsigned __int16)(v7 - 1) <= 2u );
  }
  if ( v7 != 7 )
    return 0;
  v8 = *(_QWORD *)(v6 + 40);
  v9 = *(_QWORD *)(v6 + 32);
  if ( v8 == 2 )
  {
    v10 = *(_QWORD *)(v9 + 8);
    goto LABEL_8;
  }
  v15 = 8 * v8;
  v36 = *(_QWORD *)(v6 + 48);
  v16 = (const void *)(v9 + v15);
  v17 = v15 - 8;
  v18 = (const void *)(v9 + 8);
  v39 = v41;
  v40 = 0x300000000LL;
  v19 = v17 >> 3;
  if ( (unsigned __int64)v17 > 0x18 )
  {
    na = (size_t)v18;
    v33 = v16;
    sub_16CD150(&v39, v41, v17 >> 3, 8);
    v20 = v39;
    v21 = v40;
    v19 = v17 >> 3;
    v16 = v33;
    v18 = (const void *)na;
    v22 = &v39[8 * (unsigned int)v40];
  }
  else
  {
    v20 = v41;
    v21 = 0;
    v22 = v41;
  }
  if ( v16 != v18 )
  {
    v34 = v19;
    memcpy(v22, v18, v17);
    v20 = v39;
    v21 = v40;
    LODWORD(v19) = v34;
  }
  LODWORD(v40) = v21 + v19;
  v23 = (unsigned int)(v21 + v19);
  v42 = (__int64 *)v44;
  v24 = 8 * v23;
  v43 = 0x400000000LL;
  if ( v23 > 4 )
  {
    src = v20;
    n = 8 * v23;
    v32 = v23;
    sub_16CD150(&v42, v44, v23, 8);
    LODWORD(v23) = v32;
    v24 = n;
    v20 = src;
    v28 = &v42[(unsigned int)v43];
LABEL_34:
    v35 = v23;
    memcpy(v28, v20, v24);
    LODWORD(v24) = v43;
    LODWORD(v23) = v35;
    goto LABEL_24;
  }
  if ( v24 )
  {
    v28 = (__int64 *)v44;
    goto LABEL_34;
  }
LABEL_24:
  LODWORD(v43) = v23 + v24;
  v10 = sub_14785F0(a2, &v42, v36, 0);
  if ( v42 != (__int64 *)v44 )
  {
    v37 = v10;
    _libc_free((unsigned __int64)v42);
    v10 = v37;
  }
  if ( v39 != v41 )
  {
    dest = (void *)v10;
    _libc_free((unsigned __int64)v39);
    v10 = (__int64)dest;
  }
LABEL_8:
  if ( !v10 )
    return 0;
  v11 = *(_WORD *)(v10 + 24);
  if ( a1 == v5 && v11 == 5 )
  {
    v25 = *(_QWORD **)(v10 + 32);
    if ( !*(_WORD *)(*v25 + 24LL) )
    {
      v26 = *(_QWORD *)(*v25 + 32LL);
      v27 = *(_DWORD *)(v26 + 32);
      if ( v27 <= 0x40
        && (__int64)(*(_QWORD *)(v26 + 24) << (64 - (unsigned __int8)v27)) >> (64 - (unsigned __int8)v27) == 1 )
      {
        v10 = v25[1];
        v11 = *(_WORD *)(v10 + 24);
        goto LABEL_10;
      }
    }
    return 0;
  }
LABEL_10:
  v12 = 0;
  if ( (unsigned __int16)(v11 - 1) <= 2u )
  {
    v12 = *(_QWORD *)(v10 + 40);
    v10 = *(_QWORD *)(v10 + 32);
    v11 = *(_WORD *)(v10 + 24);
  }
  if ( v11 != 10 )
    return 0;
  v13 = *(_QWORD *)(v10 - 8);
  if ( !sub_13FC1A0(a3, v13) )
    return 0;
  if ( v12 )
    return sub_14C4490(v13, a3, v12);
  return v13;
}
