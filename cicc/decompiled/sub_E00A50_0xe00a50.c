// Function: sub_E00A50
// Address: 0xe00a50
//
__int64 __fastcall sub_E00A50(__int64 a1, __int64 a2)
{
  __int64 v3; // r9
  __int64 *v5; // rsi
  __int64 result; // rax
  _QWORD *v7; // rbx
  _BYTE *v8; // rdi
  unsigned __int64 v9; // r8
  __int64 v10; // r14
  _QWORD *v11; // rax
  int v12; // edx
  _BYTE *v13; // r9
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // r14d
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 *v21; // rdx
  __int64 *v22; // rdi
  int v23; // eax
  __int64 v24; // [rsp-70h] [rbp-70h]
  _BYTE *v25; // [rsp-70h] [rbp-70h]
  int v26; // [rsp-70h] [rbp-70h]
  _BYTE *v27; // [rsp-68h] [rbp-68h] BYREF
  __int64 v28; // [rsp-60h] [rbp-60h]
  _BYTE v29[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( !a2 )
    return 0;
  if ( !sub_DFF600(a1) )
    return a1;
  v5 = (__int64 *)((*(_BYTE *)(a1 - 16) & 2) != 0);
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
  {
    if ( *(_DWORD *)(a1 - 24) <= 3u )
      return a1;
    v7 = *(_QWORD **)(a1 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) <= 3 )
      return a1;
    v7 = (_QWORD *)(a1 - 8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF) - 16);
  }
  v8 = (_BYTE *)v7[1];
  if ( v8 && (unsigned __int8)(*v8 - 5) <= 0x1Fu && !sub_DFF670((__int64)v8) )
    return a1;
  if ( a2 == -1 )
    return 0;
  if ( (_BYTE)v5 )
    v9 = *(unsigned int *)(a1 - 24);
  else
    v9 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v27 = v29;
  v10 = v9;
  v28 = 0x400000000LL;
  if ( v9 > 4 )
  {
    v5 = (__int64 *)v29;
    v26 = v9;
    sub_C8D5F0((__int64)&v27, v29, v9, 8u, v9, v3);
    v13 = v27;
    v12 = v28;
    LODWORD(v9) = v26;
    v11 = &v27[8 * (unsigned int)v28];
  }
  else
  {
    v11 = v29;
    v12 = 0;
    v13 = v29;
  }
  if ( v10 * 8 )
  {
    v14 = &v11[v10];
    do
    {
      if ( v11 )
        *v11 = *v7;
      ++v11;
      ++v7;
    }
    while ( v11 != v14 );
    v13 = v27;
    v12 = v28;
  }
  v15 = *((_QWORD *)v13 + 3);
  LODWORD(v28) = v9 + v12;
  v16 = *(_QWORD *)(v15 + 136);
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 > 0x40 )
  {
    v25 = v13;
    v23 = sub_C444A0(v16 + 24);
    v13 = v25;
    if ( v17 - v23 > 0x40 )
      goto LABEL_26;
    v18 = **(_QWORD **)(v16 + 24);
  }
  else
  {
    v18 = *(_QWORD *)(v16 + 24);
  }
  result = a1;
  if ( a2 != v18 )
  {
LABEL_26:
    v19 = sub_AD64C0(*(_QWORD *)(v16 + 8), a2, 0);
    v20 = sub_B98A20(v19, a2);
    v5 = (__int64 *)v27;
    v21 = (__int64 *)(unsigned int)v28;
    *((_QWORD *)v27 + 3) = v20;
    v22 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
      v22 = (__int64 *)*v22;
    result = sub_B9C770(v22, v5, v21, 0, 1);
    v13 = v27;
  }
  if ( v13 != v29 )
  {
    v24 = result;
    _libc_free(v13, v5);
    return v24;
  }
  return result;
}
