// Function: sub_FD53A0
// Address: 0xfd53a0
//
__int64 __fastcall sub_FD53A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // r12
  _QWORD *v19; // r13
  __int64 v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_33;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6 + 176;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F875EC )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_34;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F875EC)
      + 176;
  v12 = (_QWORD *)sub_22077B0(272);
  v13 = v12;
  if ( v12 )
  {
    *v12 = a2;
    v14 = v12 + 26;
    *(v14 - 25) = v11;
    *(v14 - 24) = v8;
    *(v14 - 23) = 0;
    *(v14 - 22) = 0;
    *(v14 - 21) = 0;
    *((_BYTE *)v14 - 160) = 0;
    *(v14 - 19) = 0;
    *(v14 - 18) = 0;
    *(v14 - 17) = 0;
    *((_DWORD *)v14 - 32) = 0;
    *(v14 - 15) = 0;
    *(v14 - 14) = 0;
    *(v14 - 13) = 0;
    *(v14 - 12) = 0;
    *(v14 - 11) = 0;
    *(v14 - 10) = 0;
    *((_DWORD *)v14 - 18) = 0;
    *(v14 - 8) = 0;
    *(v14 - 7) = 0;
    *(v14 - 6) = 0;
    *((_DWORD *)v14 - 10) = 0;
    *(v14 - 4) = 0;
    v13[23] = v14;
    v13[24] = 8;
    *((_DWORD *)v13 + 50) = 0;
    *((_BYTE *)v13 + 204) = 1;
  }
  v15 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v13;
  if ( v15 )
  {
    if ( !*(_BYTE *)(v15 + 204) )
      _libc_free(*(_QWORD *)(v15 + 184), &unk_4F875EC);
    v16 = 16LL * *(unsigned int *)(v15 + 168);
    sub_C7D6A0(*(_QWORD *)(v15 + 152), v16, 8);
    v17 = *(unsigned int *)(v15 + 136);
    if ( (_DWORD)v17 )
    {
      v18 = *(_QWORD **)(v15 + 120);
      v19 = &v18[2 * v17];
      do
      {
        if ( *v18 != -4096 && *v18 != -8192 )
        {
          v20 = v18[1];
          if ( v20 )
          {
            v21 = *(_QWORD *)(v20 + 96);
            if ( v21 != v20 + 112 )
              _libc_free(v21, v16);
            v22 = *(_QWORD *)(v20 + 24);
            if ( v22 != v20 + 40 )
              _libc_free(v22, v16);
            v16 = 168;
            j_j___libc_free_0(v20, 168);
          }
        }
        v18 += 2;
      }
      while ( v19 != v18 );
      LODWORD(v17) = *(_DWORD *)(v15 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v15 + 120), 16LL * (unsigned int)v17, 8);
    v23 = *(_QWORD *)(v15 + 88);
    if ( v23 )
      j_j___libc_free_0(v23, *(_QWORD *)(v15 + 104) - v23);
    sub_C7D6A0(*(_QWORD *)(v15 + 64), 16LL * *(unsigned int *)(v15 + 80), 8);
    j_j___libc_free_0(v15, 272);
    v13 = *(_QWORD **)(a1 + 176);
  }
  sub_CE6510(v13);
  return 0;
}
