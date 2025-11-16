// Function: sub_2FC9920
// Address: 0x2fc9920
//
void __fastcall sub_2FC9920(unsigned __int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  void (__fastcall *v13)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 928);
  *(_QWORD *)a1 = &unk_4A2C250;
  sub_C7D6A0(*(_QWORD *)(a1 + 912), v2, 8);
  if ( *(_BYTE *)(a1 + 896) )
  {
    *(_BYTE *)(a1 + 896) = 0;
    sub_FFCE90(a1 + 200, v2, v3, v4, v5, v6);
    sub_FFD870(a1 + 200, v2, v7, v8, v9, v10);
    sub_FFBC40(a1 + 200, v2);
    v11 = *(_QWORD **)(a1 + 880);
    v12 = *(_QWORD **)(a1 + 872);
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
      v12 = *(_QWORD **)(a1 + 872);
    }
    if ( v12 )
      j_j___libc_free_0((unsigned __int64)v12);
    if ( !*(_BYTE *)(a1 + 796) )
      _libc_free(*(_QWORD *)(a1 + 776));
    v15 = *(_QWORD *)(a1 + 200);
    if ( v15 != a1 + 216 )
      _libc_free(v15);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
