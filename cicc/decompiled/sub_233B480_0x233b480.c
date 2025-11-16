// Function: sub_233B480
// Address: 0x233b480
//
void __fastcall sub_233B480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  void (__fastcall *v13)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdi

  if ( *(_BYTE *)(a1 + 284) )
  {
    if ( *(_BYTE *)(a1 + 124) )
      goto LABEL_3;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 264));
    if ( *(_BYTE *)(a1 + 124) )
      goto LABEL_3;
  }
  _libc_free(*(_QWORD *)(a1 + 104));
LABEL_3:
  v6 = *(_QWORD *)(a1 + 48);
  if ( v6 )
  {
    sub_FFCE90(*(_QWORD *)(a1 + 48), a2, a3, a4, a5, a6);
    sub_FFD870(v6, a2, v7, v8, v9, v10);
    sub_FFBC40(v6, a2);
    v11 = *(_QWORD **)(v6 + 680);
    v12 = *(_QWORD **)(v6 + 672);
    if ( v11 != v12 )
    {
      do
      {
        v13 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v12[7];
        *v12 = &unk_49E5048;
        if ( v13 )
          v13(v12 + 5, v12 + 5, 3);
        *v12 = &unk_49DB368;
        v14 = v12[3];
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
          sub_BD60C0(v12 + 1);
        v12 += 9;
      }
      while ( v11 != v12 );
      v12 = *(_QWORD **)(v6 + 672);
    }
    if ( v12 )
      j_j___libc_free_0((unsigned __int64)v12);
    if ( *(_BYTE *)(v6 + 596) )
    {
      v15 = *(_QWORD *)v6;
      if ( *(_QWORD *)v6 == v6 + 16 )
      {
LABEL_17:
        j_j___libc_free_0(v6);
        return;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v6 + 576));
      v15 = *(_QWORD *)v6;
      if ( *(_QWORD *)v6 == v6 + 16 )
        goto LABEL_17;
    }
    _libc_free(v15);
    goto LABEL_17;
  }
}
