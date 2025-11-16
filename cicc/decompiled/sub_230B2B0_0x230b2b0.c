// Function: sub_230B2B0
// Address: 0x230b2b0
//
void __fastcall sub_230B2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  void (__fastcall *v14)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdi

  v6 = *(_BYTE *)(a1 + 292) == 0;
  *(_QWORD *)a1 = &unk_4A0FAF8;
  if ( v6 )
  {
    _libc_free(*(_QWORD *)(a1 + 272));
    if ( *(_BYTE *)(a1 + 132) )
      goto LABEL_3;
  }
  else if ( *(_BYTE *)(a1 + 132) )
  {
    goto LABEL_3;
  }
  _libc_free(*(_QWORD *)(a1 + 112));
LABEL_3:
  v7 = *(_QWORD *)(a1 + 56);
  if ( v7 )
  {
    sub_FFCE90(*(_QWORD *)(a1 + 56), a2, a3, a4, a5, a6);
    sub_FFD870(v7, a2, v8, v9, v10, v11);
    sub_FFBC40(v7, a2);
    v12 = *(_QWORD **)(v7 + 680);
    v13 = *(_QWORD **)(v7 + 672);
    if ( v12 != v13 )
    {
      do
      {
        v14 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v13[7];
        *v13 = &unk_49E5048;
        if ( v14 )
          v14(v13 + 5, v13 + 5, 3);
        *v13 = &unk_49DB368;
        v15 = v13[3];
        if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
          sub_BD60C0(v13 + 1);
        v13 += 9;
      }
      while ( v12 != v13 );
      v13 = *(_QWORD **)(v7 + 672);
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( *(_BYTE *)(v7 + 596) )
    {
      v16 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 == v7 + 16 )
      {
LABEL_17:
        j_j___libc_free_0(v7);
        return;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v7 + 576));
      v16 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 == v7 + 16 )
        goto LABEL_17;
    }
    _libc_free(v16);
    goto LABEL_17;
  }
}
