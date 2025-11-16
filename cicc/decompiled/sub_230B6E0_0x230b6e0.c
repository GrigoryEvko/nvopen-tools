// Function: sub_230B6E0
// Address: 0x230b6e0
//
void __fastcall sub_230B6E0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  unsigned __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rbx
  _QWORD *v14; // r15
  void (__fastcall *v15)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rdi

  v7 = *(_BYTE *)(a1 + 292) == 0;
  *(_QWORD *)a1 = &unk_4A0FAF8;
  if ( v7 )
    _libc_free(*(_QWORD *)(a1 + 272));
  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
  v8 = *(_QWORD *)(a1 + 56);
  if ( v8 )
  {
    sub_FFCE90(*(_QWORD *)(a1 + 56), a2, a3, a4, a5, a6);
    sub_FFD870(v8, a2, v9, v10, v11, v12);
    sub_FFBC40(v8, a2);
    v13 = *(_QWORD **)(v8 + 680);
    v14 = *(_QWORD **)(v8 + 672);
    if ( v13 != v14 )
    {
      do
      {
        v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v14[7];
        *v14 = &unk_49E5048;
        if ( v15 )
          v15(v14 + 5, v14 + 5, 3);
        *v14 = &unk_49DB368;
        v16 = v14[3];
        if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
          sub_BD60C0(v14 + 1);
        v14 += 9;
      }
      while ( v13 != v14 );
      v14 = *(_QWORD **)(v8 + 672);
    }
    if ( v14 )
      j_j___libc_free_0((unsigned __int64)v14);
    if ( *(_BYTE *)(v8 + 596) )
    {
      v17 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 == v8 + 16 )
      {
LABEL_19:
        j_j___libc_free_0(v8);
        goto LABEL_20;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v8 + 576));
      v17 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 == v8 + 16 )
        goto LABEL_19;
    }
    _libc_free(v17);
    goto LABEL_19;
  }
LABEL_20:
  j_j___libc_free_0(a1);
}
