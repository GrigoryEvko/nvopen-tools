// Function: sub_39CE8E0
// Address: 0x39ce8e0
//
void __fastcall sub_39CE8E0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _BYTE *v3; // rax
  bool v4; // zf
  unsigned __int64 *v6; // r8
  unsigned __int64 v7; // r9
  _BYTE *v8; // rdi
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rbx
  int v12; // r8d
  int v13; // r9d
  _BYTE *v14; // rax
  _BYTE *v15; // rdi
  unsigned __int64 v16; // rdx
  _QWORD *v17; // rcx
  __int64 v18; // rax
  unsigned __int64 *v19; // r8
  unsigned __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rbx
  const void *v23; // r15
  __int64 v24; // r13
  bool v25; // [rsp-89h] [rbp-89h] BYREF
  _BYTE *v26; // [rsp-88h] [rbp-88h] BYREF
  __int64 v27; // [rsp-80h] [rbp-80h]
  _BYTE v28[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( !a2 )
    return;
  v3 = (_BYTE *)a2[1];
  if ( !v3 )
    return;
  v4 = *a2 == 0;
  v26 = v28;
  v27 = 0x800000000LL;
  if ( v4 || *v3 != 17 )
  {
    if ( sub_39C7D40((__int64)a1, (__int64)a2) )
      goto LABEL_6;
    v25 = 0;
    sub_39CD420((__int64)a1, (__int64)a2, (__int64)&v26, &v25, v6, v7);
    if ( v25 )
    {
      v11 = sub_39CB4A0(a1, (__int64)a2);
      goto LABEL_11;
    }
    v21 = *(unsigned int *)(a3 + 8);
    v22 = (unsigned int)v27;
    v23 = v26;
    v24 = 8LL * (unsigned int)v27;
    if ( (unsigned int)v27 > (unsigned __int64)*(unsigned int *)(a3 + 12) - v21 )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), (unsigned int)v27 + v21, 8, v9, v10);
      v21 = *(unsigned int *)(a3 + 8);
    }
    if ( v24 )
    {
      memmove((void *)(*(_QWORD *)a3 + 8 * v21), v23, 8 * v22);
      LODWORD(v21) = *(_DWORD *)(a3 + 8);
    }
    *(_DWORD *)(a3 + 8) = v21 + v22;
LABEL_6:
    v8 = v26;
    if ( v26 == v28 )
      return;
    goto LABEL_7;
  }
  v11 = sub_39CC510((__int64)a1, (__int64)a2);
  if ( !v11 )
    goto LABEL_6;
  sub_39CD420((__int64)a1, (__int64)a2, (__int64)&v26, 0, v19, v20);
LABEL_11:
  v14 = v26;
  v15 = &v26[8 * (unsigned int)v27];
  if ( v15 != v26 )
  {
    do
    {
      v16 = *(_QWORD *)v14;
      *(_QWORD *)(*(_QWORD *)v14 + 40LL) = v11;
      v17 = *(_QWORD **)(v11 + 32);
      if ( v17 )
      {
        *(_QWORD *)v16 = *v17;
        **(_QWORD **)(v11 + 32) = v16 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v14 += 8;
      *(_QWORD *)(v11 + 32) = v16;
    }
    while ( v14 != v15 );
  }
  v18 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v18 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v12, v13);
    v18 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v11;
  v8 = v26;
  ++*(_DWORD *)(a3 + 8);
  if ( v8 != v28 )
LABEL_7:
    _libc_free((unsigned __int64)v8);
}
