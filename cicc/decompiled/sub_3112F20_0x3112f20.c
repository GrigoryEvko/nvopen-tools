// Function: sub_3112F20
// Address: 0x3112f20
//
__int64 __fastcall sub_3112F20(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned int v5; // r8d
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rdi
  void (__fastcall **v14)(__int64 *, _QWORD); // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rdx
  unsigned __int64 v24; // rdi
  int v25; // esi
  __int64 v26; // rax
  __int64 v27[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_QWORD *)(*a2 + 16) == *(_QWORD *)(*a2 + 8) )
  {
    v25 = 4;
    goto LABEL_35;
  }
  if ( !(unsigned __int8)sub_3112AB0(*a2) )
  {
    if ( (unsigned __int8)sub_3112AE0(*a2, (__int64)a2, v3, v4, v5) )
    {
      v18 = *a2;
      *a2 = 0;
      v19 = sub_22077B0(0x90u);
      v8 = v19;
      if ( v19 )
      {
        *(_BYTE *)(v19 + 32) = 0;
        *(_DWORD *)(v19 + 8) = 0;
        *(_QWORD *)(v19 + 24) = 0;
        *(_QWORD *)v19 = &unk_4A32AC8;
        *(_QWORD *)(v19 + 16) = v19 + 32;
        *(_QWORD *)(v19 + 48) = 0;
        v20 = sub_22077B0(0x48u);
        if ( v20 )
        {
          *(_QWORD *)(v20 + 64) = 0;
          *(_OWORD *)(v20 + 48) = 0;
          *(_QWORD *)(v20 + 16) = v20 + 64;
          *(_QWORD *)(v20 + 24) = 1;
          *(_DWORD *)(v20 + 48) = 1065353216;
          *(_QWORD *)(v20 + 56) = 0;
          *(_OWORD *)v20 = 0;
          *(_OWORD *)(v20 + 32) = 0;
        }
        v21 = *(_QWORD *)(v8 + 48);
        *(_QWORD *)(v8 + 48) = v20;
        if ( v21 )
        {
          sub_3112140(v21 + 16);
          v22 = *(_QWORD *)(v21 + 16);
          if ( v22 != v21 + 64 )
            j_j___libc_free_0(v22);
          j_j___libc_free_0(v21);
        }
        *(_QWORD *)(v8 + 56) = 0;
        v23 = (_QWORD *)sub_22077B0(0x70u);
        if ( v23 )
        {
          memset(v23, 0, 0x70u);
          v23[4] = v23 + 6;
          v23[5] = 0x100000000LL;
          v23[12] = 0x1000000000LL;
        }
        v24 = *(_QWORD *)(v8 + 56);
        *(_QWORD *)(v8 + 56) = v23;
        if ( v24 )
          sub_31128F0(v24);
        *(_QWORD *)(v8 + 64) = v18;
        *(_QWORD *)v8 = &unk_4A32B58;
        sub_C7C840(v8 + 72, v18, 1, 35);
        *(_DWORD *)(v8 + 136) = 0;
      }
      else if ( v18 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
      }
      v14 = *(void (__fastcall ***)(__int64 *, _QWORD))v8;
      goto LABEL_15;
    }
    v25 = 5;
LABEL_35:
    sub_3112880(v27, v25);
    v26 = v27[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v6 = *a2;
  *a2 = 0;
  v7 = sub_22077B0(0x68u);
  v8 = v7;
  if ( v7 )
  {
    *(_BYTE *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 8) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)v7 = &unk_4A32AC8;
    *(_QWORD *)(v7 + 16) = v7 + 32;
    *(_QWORD *)(v7 + 48) = 0;
    v9 = sub_22077B0(0x48u);
    if ( v9 )
    {
      *(_QWORD *)(v9 + 64) = 0;
      *(_OWORD *)(v9 + 48) = 0;
      *(_QWORD *)(v9 + 16) = v9 + 64;
      *(_QWORD *)(v9 + 24) = 1;
      *(_DWORD *)(v9 + 48) = 1065353216;
      *(_QWORD *)(v9 + 56) = 0;
      *(_OWORD *)v9 = 0;
      *(_OWORD *)(v9 + 32) = 0;
    }
    v10 = *(_QWORD *)(v8 + 48);
    *(_QWORD *)(v8 + 48) = v9;
    if ( v10 )
    {
      sub_3112140(v10 + 16);
      v11 = *(_QWORD *)(v10 + 16);
      if ( v11 != v10 + 64 )
        j_j___libc_free_0(v11);
      j_j___libc_free_0(v10);
    }
    *(_QWORD *)(v8 + 56) = 0;
    v12 = (_QWORD *)sub_22077B0(0x70u);
    if ( v12 )
    {
      memset(v12, 0, 0x70u);
      v12[4] = v12 + 6;
      v12[5] = 0x100000000LL;
      v12[12] = 0x1000000000LL;
    }
    v13 = *(_QWORD *)(v8 + 56);
    *(_QWORD *)(v8 + 56) = v12;
    if ( v13 )
      sub_31128F0(v13);
    *(_QWORD *)(v8 + 64) = v6;
    v14 = (void (__fastcall **)(__int64 *, _QWORD))&unk_4A32B10;
    *(_QWORD *)v8 = &unk_4A32B10;
  }
  else
  {
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    v14 = (void (__fastcall **)(__int64 *, _QWORD))MEMORY[0];
  }
LABEL_15:
  v14[2](v27, v8);
  v15 = v27[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v27[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v15;
    if ( v8 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  }
  else
  {
    v16 = *(unsigned __int8 *)(a1 + 8);
    *(_QWORD *)a1 = v8;
    *(_BYTE *)(a1 + 8) = v16 & 0xFC | 2;
  }
  return a1;
}
