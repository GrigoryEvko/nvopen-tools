// Function: sub_DEF380
// Address: 0xdef380
//
void __fastcall sub_DEF380(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  void *v6; // r9
  __int64 v7; // r8
  __int64 v8; // rbx
  void *v9; // rcx
  __int64 *v10; // r15
  __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 *v20; // rdi
  void *src; // [rsp+8h] [rbp-68h]
  void *srca; // [rsp+8h] [rbp-68h]
  __int64 *v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 v24; // [rsp+18h] [rbp-58h]
  _BYTE v25[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = a2;
  if ( sub_D92140(*(_QWORD *)(a1 + 128), a2, *(_QWORD *)(a1 + 112)) )
    return;
  v4 = *(_QWORD *)(a1 + 128);
  v23 = (__int64 *)v25;
  v5 = *(unsigned int *)(v4 + 48);
  v6 = *(void **)(v4 + 40);
  v24 = 0x400000000LL;
  v7 = 8 * v5;
  if ( v5 > 4 )
  {
    srca = v6;
    sub_C8D5F0((__int64)&v23, v25, v5, 8u, v7, (__int64)v6);
    v6 = srca;
    v7 = 8 * v5;
    v20 = &v23[(unsigned int)v24];
LABEL_16:
    a2 = (__int64)v6;
    memcpy(v20, v6, v7);
    v7 = (unsigned int)v24;
    goto LABEL_5;
  }
  if ( v7 )
  {
    v20 = (__int64 *)v25;
    goto LABEL_16;
  }
LABEL_5:
  LODWORD(v24) = v7 + v5;
  v8 = (unsigned int)(v7 + v5);
  if ( v8 + 1 > (unsigned __int64)HIDWORD(v24) )
  {
    a2 = (__int64)v25;
    sub_C8D5F0((__int64)&v23, v25, v8 + 1, 8u, v7, (__int64)v6);
    v8 = (unsigned int)v24;
  }
  v23[v8] = v2;
  v9 = *(void **)(a1 + 112);
  v10 = v23;
  v11 = (unsigned int)(v24 + 1);
  LODWORD(v24) = v24 + 1;
  src = v9;
  v12 = (_QWORD *)sub_22077B0(184);
  v17 = v12;
  if ( v12 )
  {
    a2 = (__int64)v10;
    sub_D9AF00(v12, v10, v11, (__int64)src);
  }
  v18 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 128) = v17;
  if ( v18 )
  {
    v19 = *(_QWORD *)(v18 + 40);
    if ( v19 != v18 + 56 )
      _libc_free(v19, a2);
    a2 = 184;
    j_j___libc_free_0(v18, 184);
  }
  sub_DEF2A0(a1, a2, v13, v14, v15, v16);
  if ( v23 != (__int64 *)v25 )
    _libc_free(v23, a2);
}
