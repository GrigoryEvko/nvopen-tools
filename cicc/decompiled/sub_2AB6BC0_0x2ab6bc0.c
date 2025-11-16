// Function: sub_2AB6BC0
// Address: 0x2ab6bc0
//
__int64 __fastcall sub_2AB6BC0(__int64 a1)
{
  __int64 v2; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  void (__fastcall *v18)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // r13
  unsigned __int64 v24; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 1120), 16LL * *(unsigned int *)(a1 + 1136), 8);
  v2 = *(_QWORD *)(a1 + 1048);
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 1056);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 != v4 + 40 )
          _libc_free(v5);
        j_j___libc_free_0(v4);
      }
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 1048);
  }
  if ( v3 != a1 + 1064 )
    _libc_free(v3);
  v6 = *(_QWORD *)(a1 + 1024);
  if ( v6 != a1 + 1040 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 984), 16LL * *(unsigned int *)(a1 + 1000), 8);
  v7 = 16LL * *(unsigned int *)(a1 + 968);
  sub_C7D6A0(*(_QWORD *)(a1 + 952), v7, 8);
  sub_FFCE90(a1 + 200, v7, v8, v9, v10, v11);
  sub_FFD870(a1 + 200, v7, v12, v13, v14, v15);
  sub_FFBC40(a1 + 200, v7);
  v16 = *(_QWORD **)(a1 + 880);
  v17 = *(_QWORD **)(a1 + 872);
  if ( v16 != v17 )
  {
    do
    {
      v18 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v17[7];
      *v17 = &unk_49E5048;
      if ( v18 )
        v18(v17 + 5, v17 + 5, 3);
      *v17 = &unk_49DB368;
      v19 = v17[3];
      if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        sub_BD60C0(v17 + 1);
      v17 += 9;
    }
    while ( v16 != v17 );
    v17 = *(_QWORD **)(a1 + 872);
  }
  if ( v17 )
    j_j___libc_free_0((unsigned __int64)v17);
  if ( !*(_BYTE *)(a1 + 796) )
    _libc_free(*(_QWORD *)(a1 + 776));
  v20 = *(_QWORD *)(a1 + 200);
  if ( v20 != a1 + 216 )
    _libc_free(v20);
  if ( (*(_BYTE *)(a1 + 128) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 136), 16LL * *(unsigned int *)(a1 + 144), 8);
  v21 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 72);
    v23 = &v22[7 * v21];
    do
    {
      if ( *v22 != -8192 && *v22 != -4096 )
      {
        v24 = v22[1];
        if ( (_QWORD *)v24 != v22 + 3 )
          _libc_free(v24);
      }
      v22 += 7;
    }
    while ( v23 != v22 );
    v21 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 56 * v21, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 56), 8);
}
