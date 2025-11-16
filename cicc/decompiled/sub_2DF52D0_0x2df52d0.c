// Function: sub_2DF52D0
// Address: 0x2df52d0
//
void __fastcall sub_2DF52D0(__int64 a1, __int64 a2)
{
  char v3; // di
  __int64 v4; // rax
  int v5; // edi
  __int64 v6; // rax
  unsigned __int64 v7; // r8
  void *v8; // rdi
  size_t v9; // rdx

  *(_QWORD *)a1 = 0;
  v3 = *(_BYTE *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  *(_BYTE *)(a1 + 8) = v3;
  *(_QWORD *)(a1 + 16) = v4;
  v5 = v3 & 0x3F;
  if ( v5 )
  {
    v6 = sub_2207820(4LL * (unsigned __int8)v5);
    v7 = *(_QWORD *)a1;
    v8 = (void *)v6;
    *(_QWORD *)a1 = v6;
    if ( v7 )
    {
      j_j___libc_free_0_0(v7);
      v8 = *(void **)a1;
    }
    v9 = 4LL * (*(_BYTE *)(a2 + 8) & 0x3F);
    if ( v9 )
      memmove(v8, *(const void **)a2, v9);
  }
}
