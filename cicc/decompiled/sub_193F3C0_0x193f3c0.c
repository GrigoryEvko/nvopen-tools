// Function: sub_193F3C0
// Address: 0x193f3c0
//
bool __fastcall sub_193F3C0(__int64 **a1, char a2, __m128i a3, __m128i a4)
{
  __int64 *v5; // r14
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 *v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // r12
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 *v27; // [rsp+0h] [rbp-40h] BYREF
  __int64 v28; // [rsp+8h] [rbp-38h]
  __int64 v29; // [rsp+10h] [rbp-30h] BYREF
  __int64 v30; // [rsp+18h] [rbp-28h]

  v5 = *a1;
  v6 = *(_DWORD *)a1[1];
  v7 = (*a1)[4];
  if ( v6 )
  {
    v15 = *a1[3];
    if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      v16 = *(__int64 **)(v15 - 8);
    else
      v16 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
    v17 = sub_146F1B0(v7, *v16);
    v18 = v5[4];
    v19 = (*a1)[1];
    if ( a2 )
      v8 = sub_147B0D0(v18, v17, v19, 0);
    else
      v8 = sub_14747F0(v18, v17, v19, 0);
    v14 = sub_146F1B0((*a1)[4], *a1[2]);
  }
  else
  {
    v8 = sub_146F1B0(v7, *a1[2]);
    v9 = *a1[3];
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      v10 = *(_QWORD *)(v9 - 8);
    else
      v10 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    v11 = sub_146F1B0((*a1)[4], *(_QWORD *)(v10 + 24));
    v12 = v5[4];
    v13 = (*a1)[1];
    if ( a2 )
      v14 = sub_147B0D0(v12, v11, v13, 0);
    else
      v14 = sub_14747F0(v12, v11, v13, 0);
  }
  if ( *(_BYTE *)(*a1[3] + 16) == 39 )
  {
    v25 = (_QWORD *)(*a1)[4];
    v30 = v14;
    v29 = v8;
    v27 = &v29;
    v28 = 0x200000002LL;
    v26 = sub_147EE30(v25, &v27, 0, 0, a3, a4);
    v22 = v27;
    v23 = v26;
    if ( v27 == &v29 )
      return *a1[4] == v23;
LABEL_15:
    _libc_free((unsigned __int64)v22);
    return *a1[4] == v23;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*a1[3] + 16) - 24 > 0xF )
  {
    v23 = sub_1483CF0((_QWORD *)(*a1)[4], v8, v14, a3, a4);
    return *a1[4] == v23;
  }
  v20 = (*a1)[4];
  if ( *(_BYTE *)(*a1[3] + 16) != 35 )
  {
    v23 = sub_14806B0(v20, v8, v14, 0, 0);
    return *a1[4] == v23;
  }
  v30 = v14;
  v29 = v8;
  v27 = &v29;
  v28 = 0x200000002LL;
  v21 = sub_147DD40(v20, (__int64 *)&v27, 0, 0, a3, a4);
  v22 = v27;
  v23 = (__int64)v21;
  if ( v27 != &v29 )
    goto LABEL_15;
  return *a1[4] == v23;
}
