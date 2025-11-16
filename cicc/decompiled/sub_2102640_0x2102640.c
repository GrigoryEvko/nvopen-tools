// Function: sub_2102640
// Address: 0x2102640
//
__int64 __fastcall sub_2102640(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // r9
  unsigned int v15; // r13d
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi

  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v4 = 0;
  if ( v3 != sub_1D00B10 )
    v4 = v3();
  v5 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = v4;
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4FC450C )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_29;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4FC450C);
  v9 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 240) = v8;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4FCE424 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_30;
  }
  *(_QWORD *)(a1 + 248) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(
                            *(_QWORD *)(v10 + 8),
                            &unk_4FCE424);
  v15 = *(_DWORD *)(*(_QWORD *)(a1 + 232) + 44LL);
  if ( v15 != *(_DWORD *)(a1 + 376) )
  {
    v16 = (_QWORD *)sub_2207820(176LL * v15 + 8);
    v12 = (__int64)v16;
    if ( v16 )
    {
      *v16 = v15;
      v17 = v16 + 1;
      if ( v15 )
      {
        v18 = v16 + 1;
        v12 += 176LL * v15 + 8;
        do
        {
          *v18 = 0;
          v18[4] = v18 + 6;
          v19 = v18 + 16;
          v18 += 22;
          *(v18 - 21) = 0;
          *(v18 - 19) = 0;
          *((_DWORD *)v18 - 34) = 0;
          *((_DWORD *)v18 - 33) = 4;
          *(v18 - 8) = v19;
          *((_DWORD *)v18 - 14) = 0;
          *((_DWORD *)v18 - 13) = 4;
          *((_BYTE *)v18 - 16) = 0;
          *((_BYTE *)v18 - 15) = 0;
          *((_DWORD *)v18 - 3) = 0;
          *((_DWORD *)v18 - 2) = 0;
        }
        while ( v18 != (_QWORD *)v12 );
      }
    }
    else
    {
      v17 = 0;
    }
    v20 = *(_QWORD *)(a1 + 392);
    *(_QWORD *)(a1 + 392) = v17;
    if ( v20 )
    {
      v21 = v20 + 176LL * *(_QWORD *)(v20 - 8);
      while ( v20 != v21 )
      {
        v21 -= 176;
        v22 = *(_QWORD *)(v21 + 112);
        if ( v22 != v21 + 128 )
          _libc_free(v22);
        v23 = *(_QWORD *)(v21 + 32);
        if ( v23 != v21 + 48 )
          _libc_free(v23);
      }
      j_j_j___libc_free_0_0(v20 - 8);
    }
  }
  sub_20FC9D0(a1 + 376, a1 + 264, v15, v12, v13, v14);
  ++*(_DWORD *)(a1 + 256);
  return 0;
}
