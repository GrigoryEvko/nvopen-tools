// Function: sub_2A8AF00
// Address: 0x2a8af00
//
void __fastcall sub_2A8AF00(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // r13d
  __int64 v3; // r12
  __int64 v4; // r14
  bool v5; // cc
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  unsigned __int64 v8; // rdi

  v1 = a1;
  v2 = *(_DWORD *)(a1 + 16);
  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 16) = 0;
  while ( sub_B445A0(v3, *(_QWORD *)(v1 - 24)) )
  {
    v5 = *(_DWORD *)(v1 + 16) <= 0x40u;
    *(_QWORD *)v1 = *(_QWORD *)(v1 - 24);
    if ( !v5 )
    {
      v6 = *(_QWORD *)(v1 + 8);
      if ( v6 )
        j_j___libc_free_0_0(v6);
    }
    v7 = *(_QWORD *)(v1 - 16);
    v1 -= 24;
    *(_QWORD *)(v1 + 32) = v7;
    LODWORD(v7) = *(_DWORD *)(v1 + 16);
    *(_DWORD *)(v1 + 16) = 0;
    *(_DWORD *)(v1 + 40) = v7;
  }
  v5 = *(_DWORD *)(v1 + 16) <= 0x40u;
  *(_QWORD *)v1 = v3;
  if ( !v5 )
  {
    v8 = *(_QWORD *)(v1 + 8);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  *(_QWORD *)(v1 + 8) = v4;
  *(_DWORD *)(v1 + 16) = v2;
}
