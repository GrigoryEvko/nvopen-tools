// Function: sub_1D520C0
// Address: 0x1d520c0
//
void __fastcall sub_1D520C0(__int64 a1, const char *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // [rsp+0h] [rbp-30h]
  __int64 v13; // [rsp+8h] [rbp-28h]

  v9 = *(_DWORD *)(a1 + 32);
  if ( v9 >= *(_DWORD *)(a1 + 36) )
  {
    v12 = a6;
    v13 = a5;
    sub_1D51F60(a1 + 24, 0);
    v9 = *(_DWORD *)(a1 + 32);
    a6 = v12;
    a5 = v13;
  }
  v10 = *(_QWORD *)(a1 + 24) + 56LL * v9;
  if ( v10 )
  {
    *(_QWORD *)v10 = a2;
    *(_QWORD *)(v10 + 8) = a3;
    *(_QWORD *)(v10 + 16) = a5;
    *(_QWORD *)(v10 + 24) = a6;
    *(_QWORD *)(v10 + 40) = a4;
    *(_BYTE *)(v10 + 48) = 1;
    *(_QWORD *)(v10 + 32) = &unk_49F9A38;
    v9 = *(_DWORD *)(a1 + 32);
  }
  v11 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 32) = v9 + 1;
  sub_16B7FD0(v11, a2, a3);
}
