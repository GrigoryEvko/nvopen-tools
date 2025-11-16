// Function: sub_3958AD0
// Address: 0x3958ad0
//
__int64 __fastcall sub_3958AD0(__int64 a1, __int64 a2)
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
  __int64 v13; // rdi
  _QWORD *v14; // rax
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // r12
  _QWORD *v19; // r14
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // rdi

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_29;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6 + 160;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F9920C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_28;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9920C)
      + 160;
  v12 = (_QWORD *)sub_22077B0(0x118u);
  v13 = (__int64)v12;
  if ( v12 )
  {
    *v12 = a2;
    v14 = v12 + 27;
    *(v14 - 26) = v11;
    *(v14 - 25) = v8;
    *(v14 - 24) = 0;
    *(v14 - 23) = 0;
    *(v14 - 22) = 0;
    *((_BYTE *)v14 - 168) = 0;
    *(v14 - 20) = 0;
    *(v14 - 19) = 0;
    *(v14 - 18) = 0;
    *((_DWORD *)v14 - 34) = 0;
    *(v14 - 16) = 0;
    *(v14 - 15) = 0;
    *(v14 - 14) = 0;
    *(v14 - 13) = 0;
    *(v14 - 12) = 0;
    *(v14 - 11) = 0;
    *((_DWORD *)v14 - 20) = 0;
    *(v14 - 9) = 0;
    *(v14 - 8) = 0;
    *(v14 - 7) = 0;
    *((_DWORD *)v14 - 12) = 0;
    *(_QWORD *)(v13 + 176) = 0;
    *(_QWORD *)(v13 + 184) = v14;
    *(_QWORD *)(v13 + 192) = v14;
    *(_QWORD *)(v13 + 200) = 8;
    *(_DWORD *)(v13 + 208) = 0;
  }
  v15 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v13;
  if ( v15 )
  {
    v16 = *(_QWORD *)(v15 + 192);
    if ( v16 != *(_QWORD *)(v15 + 184) )
      _libc_free(v16);
    j___libc_free_0(*(_QWORD *)(v15 + 152));
    v17 = *(unsigned int *)(v15 + 136);
    if ( (_DWORD)v17 )
    {
      v18 = *(_QWORD **)(v15 + 120);
      v19 = &v18[2 * v17];
      do
      {
        if ( *v18 != -8 && *v18 != -16 )
        {
          v20 = v18[1];
          if ( v20 )
          {
            _libc_free(*(_QWORD *)(v20 + 48));
            _libc_free(*(_QWORD *)(v20 + 24));
            j_j___libc_free_0(v20);
          }
        }
        v18 += 2;
      }
      while ( v19 != v18 );
    }
    j___libc_free_0(*(_QWORD *)(v15 + 120));
    v21 = *(_QWORD *)(v15 + 88);
    if ( v21 )
      j_j___libc_free_0(v21);
    j___libc_free_0(*(_QWORD *)(v15 + 64));
    j_j___libc_free_0(v15);
    v13 = *(_QWORD *)(a1 + 160);
  }
  sub_1C2D330(v13);
  return 0;
}
