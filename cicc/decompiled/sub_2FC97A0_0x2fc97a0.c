// Function: sub_2FC97A0
// Address: 0x2fc97a0
//
__int64 __fastcall sub_2FC97A0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  void (__fastcall *v14)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 928);
  *(_QWORD *)a1 = &unk_4A2C250;
  sub_C7D6A0(*(_QWORD *)(a1 + 912), v2, 8);
  if ( *(_BYTE *)(a1 + 896) )
  {
    *(_BYTE *)(a1 + 896) = 0;
    sub_FFCE90(a1 + 200, v2, v3, v4, v5, v6);
    sub_FFD870(a1 + 200, v2, v8, v9, v10, v11);
    sub_FFBC40(a1 + 200, v2);
    v12 = *(_QWORD **)(a1 + 880);
    v13 = *(_QWORD **)(a1 + 872);
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
      v13 = *(_QWORD **)(a1 + 872);
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( !*(_BYTE *)(a1 + 796) )
      _libc_free(*(_QWORD *)(a1 + 776));
    v16 = *(_QWORD *)(a1 + 200);
    if ( v16 != a1 + 216 )
      _libc_free(v16);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
