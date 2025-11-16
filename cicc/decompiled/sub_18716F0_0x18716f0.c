// Function: sub_18716F0
// Address: 0x18716f0
//
__int64 __fastcall sub_18716F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // r14
  unsigned __int64 v16; // rax
  unsigned int v17; // r12d
  _BYTE *v19; // r14
  __int64 *v20; // r15
  unsigned __int64 v21; // r12
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-C8h]
  __int64 v24; // [rsp+10h] [rbp-C0h]
  _BYTE *v26; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+38h] [rbp-98h]
  _BYTE v28[32]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v29; // [rsp+60h] [rbp-70h]
  __int64 v30; // [rsp+78h] [rbp-58h]
  __int64 v31; // [rsp+88h] [rbp-48h]

  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F9E06C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_42;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9E06C);
  v8 = *(__int64 **)(a1 + 8);
  v24 = v7;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F9920C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_43;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9920C);
  v11 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL) + 80LL);
  if ( v11 )
    v11 -= 24;
  v12 = sub_157EBA0(v11);
  if ( *(_BYTE *)(v12 + 16) == 26
    && (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) == 1
    && **(_QWORD **)(a2 + 32) == sub_15F4DF0(v12, 0) )
  {
    v26 = v28;
    v27 = 0x800000000LL;
    sub_13F9EC0(a2, (__int64)&v26);
    v19 = v26;
    if ( !(_DWORD)v27 )
    {
LABEL_37:
      if ( v19 != v28 )
      {
        v17 = 0;
        _libc_free((unsigned __int64)v19);
        return v17;
      }
      return 0;
    }
    v20 = (__int64 *)v26;
    v21 = (unsigned __int64)&v26[8 * (unsigned int)(v27 - 1) + 8];
    while ( *(_BYTE *)(sub_157EBA0(*v20) + 16) == 25 )
    {
      if ( (__int64 *)v21 == ++v20 )
        goto LABEL_37;
    }
    if ( v19 != v28 )
      _libc_free((unsigned __int64)v19);
  }
  v26 = v28;
  v27 = 0x800000000LL;
  sub_13F9EC0(a2, (__int64)&v26);
  if ( (_DWORD)v27 )
  {
    v13 = 8LL * (unsigned int)v27;
    v14 = 0;
    v15 = 0x40018000000001LL;
    while ( 1 )
    {
      v16 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(*(_QWORD *)&v26[v14]) + 16) - 34;
      if ( (unsigned int)v16 <= 0x36 )
      {
        if ( _bittest64(&v15, v16) )
          break;
      }
      v14 += 8;
      if ( v13 == v14 )
        goto LABEL_29;
    }
    if ( v26 != v28 )
    {
      _libc_free((unsigned __int64)v26);
      return 0;
    }
    return 0;
  }
LABEL_29:
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  v22 = *(_DWORD *)(a1 + 156);
  if ( !v22 )
    return 0;
  *(_DWORD *)(a1 + 156) = v22 - 1;
  v17 = 0;
  sub_1AC0A10(&v26, v24 + 160, a2, 0, 0, 0);
  if ( sub_1AC1F00(&v26) )
  {
    v17 = 1;
    sub_1407870(a3, a2);
    sub_1401B00(v23 + 160, (const __m128i *)a2);
  }
  if ( v30 )
    j_j___libc_free_0(v30, v31 - v30);
  j___libc_free_0(v29);
  return v17;
}
